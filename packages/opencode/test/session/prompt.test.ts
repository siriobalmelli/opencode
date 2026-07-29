import { ConfigV1 } from "@opencode-ai/core/v1/config/config"
import { SessionV1 } from "@opencode-ai/core/v1/session"
import { Database } from "@opencode-ai/core/database/database"
import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { SessionProjector } from "@opencode-ai/core/session/projector"
import { eq } from "drizzle-orm"
import { EventV2Bridge } from "@/event-v2-bridge"
import { expect } from "bun:test"
import { Cause, Deferred, Duration, Effect, Exit, Fiber, Layer } from "effect"
import path from "path"
import { fileURLToPath, pathToFileURL } from "url"
import { NamedError } from "@opencode-ai/core/util/error"
import { Agent as AgentSvc } from "../../src/agent/agent"
import { BackgroundJob } from "@/background/job"
import { Command } from "../../src/command"
import { Config } from "@/config/config"
import { LSP } from "@/lsp/lsp"
import { MCP } from "../../src/mcp"
import { Permission } from "../../src/permission"
import { Plugin } from "../../src/plugin"
import { Provider as ProviderSvc } from "@/provider/provider"
import { Env } from "../../src/env"
import { Git } from "../../src/git"
import { Image } from "../../src/image/image"

import { Question } from "../../src/question"
import { Todo } from "../../src/session/todo"
import { Session } from "@/session/session"
import { SessionMessageTable } from "@opencode-ai/core/session/sql"
import { LLM } from "../../src/session/llm"
import { MessageV2 } from "../../src/session/message-v2"
import { FSUtil } from "@opencode-ai/core/fs-util"
import { SessionCompaction } from "../../src/session/compaction"
import { SessionSummary } from "../../src/session/summary"
import { Instruction } from "../../src/session/instruction"
import { SessionProcessor } from "../../src/session/processor"
import { overflowMinimumEstimate, SessionPrompt } from "../../src/session/prompt"
import { SessionRevert } from "../../src/session/revert"
import { SessionRunState } from "../../src/session/run-state"
import { MessageID, PartID, SessionID } from "../../src/session/schema"
import { SessionStatus } from "../../src/session/status"
import { SessionV2 } from "@opencode-ai/core/session"
import { SessionExecution } from "@opencode-ai/core/session/execution"
import { Skill } from "../../src/skill"
import { SystemPrompt } from "../../src/session/system"
import { Shell } from "@opencode-ai/core/shell"
import { Snapshot } from "../../src/snapshot"
import { ToolRegistry } from "@/tool/registry"
import { Truncate } from "@/tool/truncate"
import { CrossSpawnSpawner } from "@opencode-ai/core/cross-spawn-spawner"
import { Ripgrep } from "@opencode-ai/core/ripgrep"
import { Format } from "../../src/format"
import { TestInstance } from "../fixture/fixture"
import { awaitWithTimeout, pollWithTimeout, testEffect } from "../lib/effect"
import { reply, TestLLMServer } from "../lib/llm-server"
import { RuntimeFlags } from "@/effect/runtime-flags"
import { ProviderV2 } from "@opencode-ai/core/provider"
import { ModelV2 } from "@opencode-ai/core/model"
import { LocationServiceMap, locationServiceMapLayer } from "@opencode-ai/core/location-services"

const summary = Layer.succeed(
  SessionSummary.Service,
  SessionSummary.Service.of({
    summarize: () => Effect.void,
    diff: () => Effect.succeed([]),
    computeDiff: () => Effect.succeed([]),
  }),
)

const ref = {
  providerID: ProviderV2.ID.make("test"),
  modelID: ModelV2.ID.make("test-model"),
}

function withSh<A, E, R>(fx: () => Effect.Effect<A, E, R>) {
  return Effect.acquireUseRelease(
    Effect.sync(() => {
      const prev = process.env.SHELL
      process.env.SHELL = "/bin/sh"
      Shell.preferred.reset()
      return prev
    }),
    () => fx(),
    (prev) =>
      Effect.sync(() => {
        if (prev === undefined) delete process.env.SHELL
        else process.env.SHELL = prev
        Shell.preferred.reset()
      }),
  )
}

function toolPart(parts: SessionV1.Part[]) {
  return parts.find((part): part is SessionV1.ToolPart => part.type === "tool")
}

type CompletedToolPart = SessionV1.ToolPart & { state: SessionV1.ToolStateCompleted }
type ErrorToolPart = SessionV1.ToolPart & { state: SessionV1.ToolStateError }

function completedTool(parts: SessionV1.Part[]) {
  const part = toolPart(parts)
  expect(part?.state.status).toBe("completed")
  return part?.state.status === "completed" ? (part as CompletedToolPart) : undefined
}

function errorTool(parts: SessionV1.Part[]) {
  const part = toolPart(parts)
  expect(part?.state.status).toBe("error")
  return part?.state.status === "error" ? (part as ErrorToolPart) : undefined
}

function makeMcp(instructions: MCP.ServerInstructions[] = []) {
  return Layer.succeed(
    MCP.Service,
    MCP.Service.of({
      status: () => Effect.succeed({}),
      clients: () => Effect.succeed({}),
      instructions: () => Effect.succeed(instructions),
      tools: () => Effect.succeed({}),
      prompts: () => Effect.succeed({}),
      resources: () => Effect.succeed({}),
      resourceTemplates: () => Effect.succeed({}),
      add: () => Effect.succeed({ status: { status: "disabled" as const } }),
      connect: () => Effect.void,
      disconnect: () => Effect.void,
      getPrompt: () => Effect.succeed(undefined),
      readResource: () => Effect.succeed(undefined),
      startAuth: () => Effect.die("unexpected MCP auth in prompt-effect tests"),
      authenticate: () => Effect.die("unexpected MCP auth in prompt-effect tests"),
      finishAuth: () => Effect.die("unexpected MCP auth in prompt-effect tests"),
      removeAuth: () => Effect.void,
      supportsOAuth: () => Effect.succeed(false),
      hasStoredTokens: () => Effect.succeed(false),
      getAuthStatus: () => Effect.succeed("not_authenticated" as const),
    }),
  )
}

const lsp = Layer.succeed(
  LSP.Service,
  LSP.Service.of({
    init: () => Effect.void,
    status: () => Effect.succeed([]),
    hasClients: () => Effect.succeed(false),
    touchFile: () => Effect.void,
    diagnostics: () => Effect.succeed({}),
    hover: () => Effect.succeed(undefined),
    definition: () => Effect.succeed([]),
    references: () => Effect.succeed([]),
    implementation: () => Effect.succeed([]),
    documentSymbol: () => Effect.succeed([]),
    workspaceSymbol: () => Effect.succeed([]),
    prepareCallHierarchy: () => Effect.succeed([]),
    incomingCalls: () => Effect.succeed([]),
    outgoingCalls: () => Effect.succeed([]),
  }),
)

const processorCreateStarted: Array<() => void> = []
const processorModels: Array<Parameters<SessionProcessor.Interface["create"]>[0]["model"]> = []
const blockingProcessor = Layer.succeed(
  SessionProcessor.Service,
  SessionProcessor.Service.of({
    create: (input) =>
      Effect.sync(() => {
        processorModels.push(input.model)
        processorCreateStarted.shift()?.()
      }).pipe(Effect.andThen(Effect.never)),
  }),
)

const runtimeFlags = RuntimeFlags.layer({ experimentalEventSystem: true })

const testLLMServerNode = LayerNode.make({ service: TestLLMServer, layer: TestLLMServer.layer, deps: [] })

const promptRoot = LayerNode.group([
  SessionPrompt.node,
  Session.node,
  SessionProjector.node,
  MessageV2.node,
  Snapshot.node,
  LLM.node,
  Env.node,
  AgentSvc.node,
  Command.node,
  Permission.node,
  Plugin.node,
  Config.node,
  ProviderSvc.node,
  LSP.node,
  MCP.node,
  FSUtil.node,
  BackgroundJob.node,
  SessionStatus.node,
  SessionRunState.node,
  Database.node,
  EventV2Bridge.node,
  Question.node,
  Todo.node,
  ToolRegistry.node,
  Skill.node,
  Git.node,
  Ripgrep.node,
  Format.node,
  Truncate.node,
  SessionProcessor.node,
  Image.node,
  SessionCompaction.node,
  SessionRevert.node,
  Instruction.node,
  SystemPrompt.node,
  CrossSpawnSpawner.node,
  RuntimeFlags.node,
])

function makePrompt(input?: { mcpInstructions?: MCP.ServerInstructions[]; processor?: "blocking" }) {
  const replacements = [
    [SessionSummary.node, summary],
    [LSP.node, lsp],
    [MCP.node, makeMcp(input?.mcpInstructions)],
    [RuntimeFlags.node, runtimeFlags],
  ] as const
  if (input?.processor === "blocking") {
    return LayerNode.compile(promptRoot, [...replacements, [SessionProcessor.node, blockingProcessor]])
  }
  return LayerNode.compile(promptRoot, replacements)
}

function makeHttp(input?: { mcpInstructions?: MCP.ServerInstructions[]; processor?: "blocking" }) {
  const root = LayerNode.group([promptRoot, testLLMServerNode])
  const replacements = [
    [SessionSummary.node, summary],
    [LSP.node, lsp],
    [MCP.node, makeMcp(input?.mcpInstructions)],
    [RuntimeFlags.node, runtimeFlags],
  ] as const
  if (input?.processor === "blocking") {
    return LayerNode.compile(root, [...replacements, [SessionProcessor.node, blockingProcessor]])
  }
  return LayerNode.compile(root, replacements)
}

function makeHttpNoLLMServer(input?: { mcpInstructions?: MCP.ServerInstructions[]; processor?: "blocking" }) {
  return makePrompt(input)
}

const it = testEffect(makeHttp())
const overflow = it
const noLLMServer = testEffect(makeHttpNoLLMServer())
const raceNoLLMServer = testEffect(makeHttpNoLLMServer({ processor: "blocking" }))
const overflowCompactionCalls = new Map<SessionID, Array<Parameters<SessionCompaction.Interface["create"]>[0]>>()
const overflowCompactions = (sessionID: SessionID) => overflowCompactionCalls.get(sessionID) ?? []
const overflowCompactionSpy = Layer.succeed(
  SessionCompaction.Service,
  SessionCompaction.Service.of({
    isOverflow: () => Effect.succeed(true),
    prune: () => Effect.void,
    process: () => Effect.succeed("continue"),
    create: (input) =>
      Effect.sync(() => {
        overflowCompactionCalls.set(input.sessionID, [...overflowCompactions(input.sessionID), input])
      }).pipe(Effect.andThen(Effect.die("overflow compaction created"))),
  }),
)
const overflowHarness = testEffect(
  LayerNode.compile(LayerNode.group([promptRoot, testLLMServerNode]), [
    [SessionSummary.node, summary],
    [LSP.node, lsp],
    [MCP.node, makeMcp()],
    [RuntimeFlags.node, runtimeFlags],
    [SessionCompaction.node, overflowCompactionSpy],
  ]),
)
const cappedOverflowHarness = testEffect(
  LayerNode.compile(LayerNode.group([promptRoot, testLLMServerNode]), [
    [SessionSummary.node, summary],
    [LSP.node, lsp],
    [MCP.node, makeMcp()],
    [RuntimeFlags.node, RuntimeFlags.layer({ experimentalEventSystem: true, outputTokenMax: 5 })],
    [SessionCompaction.node, overflowCompactionSpy],
  ]),
)
const withMcpInstructions = testEffect(
  makeHttp({
    mcpInstructions: [
      {
        name: "guide-server",
        instructions: "Use lookup before mutate.",
        tools: ["guide-server_lookup"],
      },
    ],
  }),
)
const unix = process.platform !== "win32" ? it.instance : it.instance.skip
const unixNoLLMServer = process.platform !== "win32" ? noLLMServer.instance : noLLMServer.instance.skip

noLLMServer.effect("overflow minimum stringify defects succeed with no estimate", () =>
  Effect.gen(function* () {
    const messages: { self?: unknown } = {}
    messages.self = messages
    expect(yield* overflowMinimumEstimate(messages, () => 100)).toBeUndefined()
    expect(
      yield* overflowMinimumEstimate([], () => {
        throw new Error("unavailable capacity")
      }),
    ).toBeUndefined()
  }),
)

// Config that registers a custom "test" provider with a "test-model" model
// so provider model lookup succeeds inside the loop.
const cfg = {
  provider: {
    test: {
      name: "Test",
      id: "test",
      env: [],
      npm: "@ai-sdk/openai-compatible",
      models: {
        "test-model": {
          id: "test-model",
          name: "Test Model",
          attachment: false,
          reasoning: false,
          temperature: false,
          tool_call: true,
          release_date: "2025-01-01",
          limit: { context: 100000, output: 10000 },
          cost: { input: 0, output: 0 },
          options: {},
        },
      },
      options: {
        apiKey: "test-key",
        baseURL: "http://localhost:1/v1",
      },
    },
  },
}

const routedCfg = {
  ...cfg,
  provider: {
    ...cfg.provider,
    fallback: {
      ...cfg.provider.test,
      id: "fallback",
      name: "Fallback",
      models: {
        "fallback-model": { ...cfg.provider.test.models["test-model"], id: "fallback-model", name: "Fallback Model" },
      },
      options: { ...cfg.provider.test.options },
    },
  },
}

const routedNoHttpCfg = {
  ...routedCfg,
  provider: {
    ...routedCfg.provider,
    fallback: {
      ...routedCfg.provider.fallback,
      models: {
        "fallback-model": {
          ...cfg.provider.test.models["test-model"],
          id: "fallback-model",
          reasoning: true,
          variants: { high: {} },
        },
        "second-model": {
          ...cfg.provider.test.models["test-model"],
          id: "second-model",
          reasoning: true,
        },
      },
    },
  },
}

function overflowProviderCfg(url: string) {
  const model = cfg.provider.test.models["test-model"]
  return {
    ...routedProviderCfg(url),
    compaction: { reserved: 20 },
    provider: {
      ...routedProviderCfg(url).provider,
      test: {
        ...routedProviderCfg(url).provider.test,
        models: {
          "test-model": { ...model, limit: { context: 120, output: 20 } },
        },
      },
      fallback: {
        ...routedProviderCfg(url).provider.fallback,
        models: {
          "fallback-small": { ...model, id: "fallback-small", limit: { context: 80, output: 20 } },
          "fallback-bound": { ...model, id: "fallback-bound", limit: { context: 120, output: 20 } },
          "fallback-large": {
            ...model,
            id: "fallback-large",
            limit: { context: 240, output: 20 },
            variants: { high: {} },
          },
        },
      },
    },
  }
}

function providerCfg(url: string) {
  return {
    ...cfg,
    provider: {
      ...cfg.provider,
      test: {
        ...cfg.provider.test,
        options: {
          ...cfg.provider.test.options,
          baseURL: url,
        },
      },
    },
  }
}

function routedProviderCfg(url: string) {
  return {
    ...providerCfg(url),
    agent: { build: { model: "test/test-model" } },
    provider: {
      ...providerCfg(url).provider,
      fallback: {
        ...cfg.provider.test,
        id: "fallback",
        name: "Fallback",
        models: {
          "fallback-model": { ...cfg.provider.test.models["test-model"], id: "fallback-model", name: "Fallback Model" },
        },
        options: { ...cfg.provider.test.options, baseURL: url },
      },
    },
  }
}

function stickyProviderCfg(url: string) {
  return {
    ...routedProviderCfg(url),
    agent: {
      ...routedProviderCfg(url).agent,
      sticky: { description: "Test agent without a configured model" },
    },
    provider: {
      ...routedProviderCfg(url).provider,
      fallback: {
        ...routedProviderCfg(url).provider.fallback,
        models: {
          "fallback-model": {
            ...cfg.provider.test.models["test-model"],
            id: "fallback-model",
            variants: { high: {} },
          },
        },
      },
    },
  }
}

const writeText = Effect.fn("test.writeText")(function* (file: string, text: string) {
  const fs = yield* FSUtil.Service
  yield* fs.writeWithDirs(file, text)
})

function protocolEmptyCounterPlugin(counter: string) {
  return [
    "export default async () => ({",
    '  "chat.provider.failure": async (input) => {',
    '    if (input.failure !== "protocol_empty") return',
    `    const file = Bun.file(${JSON.stringify(counter)})`,
    "    const count = (await file.exists()) ? Number(await file.text()) : 0",
    `    await Bun.write(${JSON.stringify(counter)}, String(count + 1))`,
    "  },",
    "})",
  ].join("\n")
}

const protocolEmptyInvocationCount = (counter: string) =>
  Effect.promise(() => Bun.file(counter).text()).pipe(Effect.map(Number))

function overflowPlugin(input: { capture: string; action?: "fallback" | "stop"; models?: readonly unknown[] }) {
  return [
    "export default async () => ({",
    '  "chat.provider.failure": async (input, output) => {',
    '    if (input.failure !== "overflow") return',
    `    await Bun.write(${JSON.stringify(input.capture)}, JSON.stringify(input))`,
    ...(input.action ? [`    output.action = ${JSON.stringify(input.action)}`] : []),
    ...(input.models ? [`    output.models.push(...${JSON.stringify(input.models)})`] : []),
    "  },",
    "})",
  ].join("\n")
}

const chatMessageModelPlugin = [
  "export default async () => {",
  "  let routed = false",
  "  return {",
  '  "chat.message": (_input, output) => {',
  "    if (routed) return",
  "    routed = true",
  '    output.message.agent = "sticky"',
  '    output.message.model = { providerID: "fallback", modelID: "fallback-model", variant: "high" }',
  "  },",
  "  }",
  "}",
].join("\n")

const writeConfig = Effect.fn("test.writeConfig")(function* (dir: string, config: Partial<ConfigV1.Info>) {
  yield* writeText(
    path.join(dir, "opencode.json"),
    JSON.stringify({ $schema: "https://opencode.ai/config.json", ...config }),
  )
})

const useServerConfig = Effect.fn("test.useServerConfig")(function* (config: (url: string) => Partial<ConfigV1.Info>) {
  const { directory: dir } = yield* TestInstance
  const llm = yield* TestLLMServer
  yield* writeConfig(dir, config(llm.url))
  const service = yield* Config.Service
  yield* service.invalidate()
  yield* service.get()
  return { dir, llm }
})

// Wait for a session's runner to enter a busy state. SessionStatus is flipped
// inside Runner.startShell's serialized transition, so cancel can't no-op once
// we observe it.
const waitForBusy = (sessionID: SessionID, duration: Duration.Input = "2 seconds") =>
  pollWithTimeout(
    Effect.gen(function* () {
      const status = yield* SessionStatus.Service
      const s = yield* status.get(sessionID)
      return s.type === "busy" ? (true as const) : undefined
    }),
    `session ${sessionID} never became busy`,
    duration,
  )

const hasBash = Effect.sync(() => Bun.which("bash") !== null)

const deferredAsPromise = <A>(deferred: Deferred.Deferred<A>): PromiseLike<A> => ({
  then: (onfulfilled, onrejected) => {
    Effect.runFork(
      Deferred.await(deferred).pipe(
        Effect.match({
          onFailure: (error) => {
            onrejected?.(error)
          },
          onSuccess: (value) => {
            onfulfilled?.(value)
          },
        }),
      ),
    )
    return deferredAsPromise(deferred) as PromiseLike<never>
  },
})

function defer<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void
  const promise = new Promise<T>((done) => {
    resolve = done
  })
  return { promise, resolve }
}

const succeedVoid = (deferred: Deferred.Deferred<void>) => {
  Effect.runSync(Deferred.succeed(deferred, void 0).pipe(Effect.ignore))
}

const user = Effect.fn("test.user")(function* (sessionID: SessionID, text: string) {
  const session = yield* Session.Service
  const msg = yield* session.updateMessage({
    id: MessageID.ascending(),
    role: "user",
    sessionID,
    agent: "build",
    model: ref,
    time: { created: Date.now() },
  })
  yield* session.updatePart({
    id: PartID.ascending(),
    messageID: msg.id,
    sessionID,
    type: "text",
    text,
  })
  return msg
})

const seed = Effect.fn("test.seed")(function* (sessionID: SessionID, opts?: { finish?: string }) {
  const session = yield* Session.Service
  const msg = yield* user(sessionID, "hello")
  const assistant: SessionV1.Assistant = {
    id: MessageID.ascending(),
    role: "assistant",
    parentID: msg.id,
    sessionID,
    mode: "build",
    agent: "build",
    cost: 0,
    path: { cwd: "/tmp", root: "/tmp" },
    tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
    modelID: ref.modelID,
    providerID: ref.providerID,
    time: { created: Date.now() },
    ...(opts?.finish ? { finish: opts.finish } : {}),
  }
  yield* session.updateMessage(assistant)
  yield* session.updatePart({
    id: PartID.ascending(),
    messageID: assistant.id,
    sessionID,
    type: "text",
    text: "hi there",
  })
  return { user: msg, assistant }
})

const routedHandoff = (userMessageID: MessageID, status: "pending" | "applied" = "pending") => ({
  id: "handoff-test",
  status,
  failure: "rate_limit" as const,
  from: ref,
  next: { providerID: ProviderV2.ID.make("fallback"), modelID: ModelV2.ID.make("fallback-model") },
  userMessageID,
})

const seedRoutedHandoff = Effect.fn("test.seedRoutedHandoff")(function* (
  sessionID: SessionID,
  input?: {
    status?: "pending" | "applied"
    tagged?: boolean
    terminal?: boolean
    cancelled?: boolean
    differentTagged?: boolean
    userMarker?: boolean
    finish?: string
    tail?: boolean
  },
) {
  const sessions = yield* Session.Service
  const seeded = yield* seed(sessionID, { finish: "error" })
  const user = seeded.user as SessionV1.User
  const handoff = routedHandoff(seeded.user.id, input?.status)
  if (input?.userMarker !== false) {
    user.routedHandoff = handoff
    yield* sessions.updateMessage(user)
  }
  seeded.assistant.routedHandoff = handoff
  yield* sessions.updateMessage(seeded.assistant)
  if (input?.tagged) {
    const successor: SessionV1.Assistant = {
      ...seeded.assistant,
      id: MessageID.ascending(),
      modelID: handoff.next.modelID,
      providerID: handoff.next.providerID,
      routedHandoffID: input.differentTagged ? "different-handoff" : handoff.id,
      time: { created: Date.now(), ...(input.terminal || input.cancelled ? { completed: Date.now() } : {}) },
      ...(input.terminal
        ? { finish: "stop" }
        : {
            finish: input.finish,
            ...(input.cancelled
              ? {
                  error: MessageV2.fromError(new DOMException("Aborted", "AbortError"), {
                    providerID: ref.providerID,
                    aborted: true,
                  }),
                }
              : { error: undefined }),
          }),
    }
    yield* sessions.updateMessage(successor)
    if (input.tail) {
      yield* sessions.updateMessage({
        ...successor,
        id: MessageID.ascending(),
        routedHandoffID: undefined,
        time: { created: Date.now() },
        finish: undefined,
        error: undefined,
      })
    }
  }
  return { ...seeded, user, handoff }
})

const addSubtask = (sessionID: SessionID, messageID: MessageID, model = ref) =>
  Effect.gen(function* () {
    const session = yield* Session.Service
    yield* session.updatePart({
      id: PartID.ascending(),
      messageID,
      sessionID,
      type: "subtask",
      prompt: "look into the cache key path",
      description: "inspect bug",
      agent: "general",
      model,
    })
  })

const boot = Effect.fn("test.boot")(function* (input?: { title?: string }) {
  const config = yield* Config.Service
  const prompt = yield* SessionPrompt.Service
  const run = yield* SessionRunState.Service
  const sessions = yield* Session.Service
  yield* config.get()
  const chat = yield* sessions.create(input ?? { title: "Pinned" })
  return { prompt, run, sessions, chat }
})

it.instance(
  "persists chat.message model mutations on the user and session",
  () =>
    Effect.gen(function* () {
      const { directory } = yield* TestInstance
      const plugin = path.join(directory, "sticky-model-plugin.ts")
      yield* writeText(plugin, chatMessageModelPlugin)
      const { llm } = yield* useServerConfig((url) => ({
        ...stickyProviderCfg(url),
        plugin: [pathToFileURL(plugin).href],
      }))
      const events = yield* EventV2Bridge.Service
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Sticky plugin model" })
      const original = sessions.setAgentModel
      let modelUpdates = 0
      ;(sessions as { setAgentModel: typeof sessions.setAgentModel }).setAgentModel = (input) => {
        modelUpdates++
        return original(input)
      }
      yield* Effect.addFinalizer(() =>
        Effect.sync(() => {
          ;(sessions as { setAgentModel: typeof sessions.setAgentModel }).setAgentModel = original
        }),
      )
      const updates: Array<Session.Info["model"]> = []
      const off = yield* events.listen((event) => {
        if (event.type === Session.Event.Updated.type) {
          const data = event.data as typeof Session.Event.Updated.data.Type
          if (data.sessionID === chat.id) updates.push(data.info.model)
        }
        return Effect.void
      })

      const message = yield* prompt.prompt({
        sessionID: chat.id,
        agent: "sticky",
        model: ref,
        noReply: true,
        parts: [{ type: "text", text: "hello" }],
      })
      yield* off

      expect(modelUpdates).toBe(1)
      expect(updates.some((model) => model?.providerID === ref.providerID && model.id === ref.modelID)).toBe(false)
      expect(message.info.role).toBe("user")
      if (message.info.role === "user") {
        expect(message.info.agent).toBe("sticky")
        expect(message.info.model).toEqual({
          providerID: ProviderV2.ID.make("fallback"),
          modelID: ModelV2.ID.make("fallback-model"),
          variant: "high",
        })
      }
      expect(yield* sessions.get(chat.id)).toMatchObject({
        agent: "sticky",
        model: {
          providerID: ProviderV2.ID.make("fallback"),
          id: ModelV2.ID.make("fallback-model"),
          variant: "high",
        },
      })
      expect(yield* llm.hits).toHaveLength(0)
    }),
  10_000,
)

it.instance("uses the chat.message model mutation on the next implicit turn", () =>
  Effect.gen(function* () {
    const { directory } = yield* TestInstance
    const plugin = path.join(directory, "sticky-next-turn-plugin.ts")
    yield* writeText(plugin, chatMessageModelPlugin)
    yield* useServerConfig((url) => ({ ...stickyProviderCfg(url), plugin: [pathToFileURL(plugin).href] }))
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const chat = yield* sessions.create({ title: "Sticky next turn" })

    yield* prompt.prompt({
      sessionID: chat.id,
      agent: "sticky",
      model: ref,
      noReply: true,
      parts: [{ type: "text", text: "first" }],
    })
    const next = yield* prompt.prompt({
      sessionID: chat.id,
      agent: "sticky",
      noReply: true,
      parts: [{ type: "text", text: "second" }],
    })

    expect(next.info.role).toBe("user")
    if (next.info.role === "user") {
      expect(next.info.model).toEqual({
        providerID: ProviderV2.ID.make("fallback"),
        modelID: ModelV2.ID.make("fallback-model"),
        variant: "high",
      })
    }
  }),
)

it.instance("uses the persisted chat.message model when history is unavailable", () =>
  Effect.gen(function* () {
    const { directory } = yield* TestInstance
    const plugin = path.join(directory, "sticky-restart-plugin.ts")
    yield* writeText(plugin, chatMessageModelPlugin)
    yield* useServerConfig((url) => ({ ...stickyProviderCfg(url), plugin: [pathToFileURL(plugin).href] }))
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const chat = yield* sessions.create({ title: "Sticky restart" })

    const first = yield* prompt.prompt({
      sessionID: chat.id,
      agent: "sticky",
      model: ref,
      noReply: true,
      parts: [{ type: "text", text: "first" }],
    })
    yield* sessions.removeMessage({ sessionID: chat.id, messageID: first.info.id })
    const next = yield* prompt.prompt({
      sessionID: chat.id,
      agent: "sticky",
      noReply: true,
      parts: [{ type: "text", text: "after restart" }],
    })

    expect(next.info.role).toBe("user")
    if (next.info.role === "user") {
      expect(next.info.model).toEqual({
        providerID: ProviderV2.ID.make("fallback"),
        modelID: ModelV2.ID.make("fallback-model"),
        variant: "high",
      })
    }
  }),
)

it.instance("does not emit a redundant session update for an unchanged chat.message hook", () =>
  Effect.gen(function* () {
    const { directory } = yield* TestInstance
    const plugin = path.join(directory, "unchanged-message-plugin.ts")
    yield* writeText(plugin, 'export default async () => ({ "chat.message": () => {} })')
    yield* useServerConfig((url) => ({ ...stickyProviderCfg(url), plugin: [pathToFileURL(plugin).href] }))
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const events = yield* EventV2Bridge.Service
    const chat = yield* sessions.create({
      title: "Unchanged plugin model",
      agent: "sticky",
      model: {
        providerID: ProviderV2.ID.make("fallback"),
        id: ModelV2.ID.make("fallback-model"),
        variant: "high",
      },
    })
    const original = sessions.setAgentModel
    let updates = 0
    ;(sessions as { setAgentModel: typeof sessions.setAgentModel }).setAgentModel = (input) => {
      updates++
      return original(input)
    }
    yield* Effect.addFinalizer(() =>
      Effect.sync(() => {
        ;(sessions as { setAgentModel: typeof sessions.setAgentModel }).setAgentModel = original
      }),
    )
    const modelUpdates: Session.Info["model"][] = []
    const off = yield* events.listen((event) => {
      if (event.type !== Session.Event.Updated.type) return Effect.void
      const data = event.data as typeof Session.Event.Updated.data.Type
      if (data.sessionID === chat.id) modelUpdates.push(data.info.model)
      return Effect.void
    })

    yield* prompt.prompt({
      sessionID: chat.id,
      agent: "sticky",
      noReply: true,
      parts: [{ type: "text", text: "hello" }],
    })
    yield* off

    expect(updates).toBe(0)
    expect(modelUpdates).toEqual([
      { providerID: ProviderV2.ID.make("fallback"), id: ModelV2.ID.make("fallback-model"), variant: "high" },
    ])
  }),
)

it.instance("keeps the session model unchanged when no chat.message plugin is configured", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig(stickyProviderCfg)
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const chat = yield* sessions.create({
      title: "No plugin model",
      agent: "sticky",
      model: {
        providerID: ProviderV2.ID.make("fallback"),
        id: ModelV2.ID.make("fallback-model"),
        variant: "high",
      },
    })

    const message = yield* prompt.prompt({
      sessionID: chat.id,
      agent: "sticky",
      noReply: true,
      parts: [{ type: "text", text: "hello" }],
    })

    expect(message.info.role).toBe("user")
    if (message.info.role === "user") {
      expect(message.info.model).toEqual({
        providerID: ProviderV2.ID.make("fallback"),
        modelID: ModelV2.ID.make("fallback-model"),
        variant: "high",
      })
    }
    const explicit = yield* prompt.prompt({
      sessionID: chat.id,
      agent: "sticky",
      model: ref,
      noReply: true,
      parts: [{ type: "text", text: "explicit" }],
    })

    expect(explicit.info.role).toBe("user")
    if (explicit.info.role === "user") expect(explicit.info.model).toEqual(ref)
    expect((yield* sessions.get(chat.id)).model).toEqual({
      providerID: ref.providerID,
      id: ref.modelID,
      variant: "default",
    })
    expect(yield* llm.hits).toHaveLength(0)
  }),
)

// Loop semantics

noLLMServer.instance(
  "loop exits immediately when last assistant has stop finish",
  () =>
    Effect.gen(function* () {
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Pinned" })
      yield* seed(chat.id, { finish: "stop" })

      const result = yield* prompt.loop({ sessionID: chat.id })
      expect(result.info.role).toBe("assistant")
      if (result.info.role === "assistant") expect(result.info.finish).toBe("stop")
    }),
  { config: cfg },
)

for (const [name, first] of [
  ["empty terminal response", () => reply().stop()],
  ["whitespace-only terminal response", () => reply().text(" \n\t ").stop()],
  ["reasoning-only terminal response", () => reply().reason("internal").stop()],
] as const) {
  it.instance(
    `routes ${name} to one tagged fallback successor`,
    () =>
      Effect.gen(function* () {
        const { directory } = yield* TestInstance
        const plugin = path.join(directory, "protocol-empty-fallback.ts")
        yield* writeText(
          plugin,
          [
            "export default async () => ({",
            '  "chat.provider.failure": (input, output) => {',
            '    if (input.failure !== "protocol_empty") return',
            '    output.action = "fallback"',
            '    output.models.push({ providerID: "fallback", modelID: "fallback-model" })',
            "  },",
            "})",
          ].join("\n"),
        )
        const { llm } = yield* useServerConfig((url) => ({
          ...routedProviderCfg(url),
          plugin: [pathToFileURL(plugin).href],
        }))
        const events = yield* EventV2Bridge.Service
        const prompt = yield* SessionPrompt.Service
        const sessions = yield* Session.Service
        const chat = yield* sessions.create({ title: name })
        const statuses: string[] = []
        const off = yield* events.listen((event) => {
          if (event.type !== SessionStatus.Event.Status.type) return Effect.void
          const data = event.data as typeof SessionStatus.Event.Status.data.Type
          if (data.sessionID === chat.id) statuses.push(data.status.type)
          return Effect.void
        })
        yield* llm.push(first(), reply().text("fallback reply").stop())

        const result = yield* awaitWithTimeout(
          prompt.prompt({ sessionID: chat.id, agent: "build", parts: [{ type: "text", text: "route" }] }),
          `timed out routing ${name}`,
          "10 seconds",
        )
        const messages = yield* sessions.messages({ sessionID: chat.id })
        yield* off
        const users = messages.filter((message) => message.info.role === "user")
        const assistants = messages.filter(
          (message): message is SessionV1.WithParts & { info: SessionV1.Assistant } =>
            message.info.role === "assistant",
        )
        const failed = assistants.filter((message) => message.info.routedHandoff?.failure === "protocol_empty")
        const successor = assistants.filter((message) => Boolean(message.info.routedHandoffID))

        expect(users).toHaveLength(1)
        expect(assistants).toHaveLength(2)
        expect(failed).toHaveLength(1)
        expect(failed[0]?.parts.filter((part) => part.type === "text").every((part) => part.text.trim() === "")).toBe(
          true,
        )
        expect(successor).toHaveLength(1)
        expect(result.info.id).toBe(successor[0]?.info.id)
        expect(successor[0]?.info.parentID).toBe(users[0]?.info.id)
        expect(successor[0]?.info.routedHandoffID).toBe(failed[0]?.info.routedHandoff?.id)
        expect(failed[0]?.info.finish).toBe("error")
        expect(failed[0]?.info.error).toBeDefined()
        expect(failed[0]?.info.routedHandoff?.userMessageID).toBe(users[0]?.info.id)
        expect(successor[0]?.info.variant).toBeUndefined()
        expect(users[0]?.info).toMatchObject({
          model: { providerID: "fallback", modelID: "fallback-model" },
          routedHandoff: {
            id: failed[0]?.info.routedHandoff?.id,
            status: "applied",
            userMessageID: users[0]?.info.id,
          },
        })
        expect((yield* sessions.get(chat.id)).model?.variant).toBeUndefined()
        expect(statuses.at(-1)).toBe("idle")
        expect(statuses.slice(0, -1)).not.toContain("idle")
        expect(yield* llm.hits).toHaveLength(2)
      }),
    { config: () => routedCfg },
    20_000,
  )
}

it.instance(
  "stops a handled empty response with a stable redacted diagnostic",
  () =>
    Effect.gen(function* () {
      const { directory } = yield* TestInstance
      const plugin = path.join(directory, "protocol-empty-stop.ts")
      yield* writeText(
        plugin,
        'export default async () => ({ "chat.provider.failure": (input, output) => { if (input.failure === "protocol_empty") output.action = "stop" } })',
      )
      const { llm } = yield* useServerConfig((url) => ({
        ...routedProviderCfg(url),
        plugin: [pathToFileURL(plugin).href],
      }))
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Empty stop" })
      yield* llm.push(reply().stop())

      yield* prompt.prompt({ sessionID: chat.id, agent: "build", parts: [{ type: "text", text: "route" }] })
      const assistants = (yield* sessions.messages({ sessionID: chat.id })).filter(
        (message): message is SessionV1.WithParts & { info: SessionV1.Assistant } => message.info.role === "assistant",
      )
      expect(assistants).toHaveLength(1)
      expect(JSON.stringify(assistants[0]?.info.error)).toContain("Routed empty-response failure: routing stopped")
      expect(JSON.stringify(assistants[0]?.info.error)).not.toContain("test-key")
      expect(yield* llm.hits).toHaveLength(1)
    }),
  { config: () => routedCfg },
  10_000,
)

it.instance(
  "stops when the empty-response hook throws without native continuation",
  () =>
    Effect.gen(function* () {
      const plugin = yield* Plugin.Service
      const original = plugin.triggerProviderFailure
      ;(plugin as { triggerProviderFailure: typeof plugin.triggerProviderFailure }).triggerProviderFailure = (() =>
        Effect.fail("provider failure hook exploded")) as Plugin.Interface["triggerProviderFailure"]
      yield* Effect.addFinalizer(() =>
        Effect.sync(() => {
          ;(plugin as { triggerProviderFailure: typeof plugin.triggerProviderFailure }).triggerProviderFailure =
            original
        }),
      )
      const { llm } = yield* useServerConfig(routedProviderCfg)
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Empty hook throw" })
      yield* llm.push(reply().stop(), reply().text("must not run").stop())

      yield* prompt.prompt({ sessionID: chat.id, agent: "build", parts: [{ type: "text", text: "route" }] })

      const assistants = (yield* sessions.messages({ sessionID: chat.id })).filter(
        (message): message is SessionV1.WithParts & { info: SessionV1.Assistant } => message.info.role === "assistant",
      )
      expect(assistants).toHaveLength(1)
      expect(JSON.stringify(assistants[0]?.info.error)).toContain("Routed empty-response failure: routing stopped")
      expect(yield* llm.hits).toHaveLength(1)
    }),
  { config: () => routedCfg },
  10_000,
)

it.instance(
  "keeps an unhandled empty response native and unmarked",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(providerCfg)
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Unhandled empty" })
      yield* llm.push(reply().stop())

      const result = yield* prompt.prompt({
        sessionID: chat.id,
        agent: "build",
        parts: [{ type: "text", text: "native" }],
      })
      expect(result.info.role).toBe("assistant")
      if (result.info.role === "assistant") {
        expect(result.info.error).toBeUndefined()
        expect(result.info.routedHandoff).toBeUndefined()
      }
      expect(yield* llm.hits).toHaveLength(1)
    }),
  { config: () => routedCfg },
  10_000,
)

for (const [name, models] of [
  ["same model", [{ providerID: "test", modelID: "test-model" }]],
  [
    "duplicate models",
    [
      { providerID: "fallback", modelID: "fallback-model" },
      { providerID: "fallback", modelID: "fallback-model" },
    ],
  ],
  ["empty models", []],
  ["unknown provider", [{ providerID: "missing", modelID: "fallback-model" }]],
  ["unknown model", [{ providerID: "fallback", modelID: "missing-model" }]],
  ["invalid variant", [{ providerID: "fallback", modelID: "fallback-model", variant: "missing" }]],
] as const) {
  it.instance(
    `stops an empty response on ${name} fallback proposal`,
    () =>
      Effect.gen(function* () {
        const { directory } = yield* TestInstance
        const plugin = path.join(directory, "protocol-empty-invalid.ts")
        yield* writeText(
          plugin,
          `export default async () => ({ "chat.provider.failure": (input, output) => { if (input.failure === "protocol_empty") { output.action = "fallback"; output.models.push(...${JSON.stringify(models)}) } } })`,
        )
        const { llm } = yield* useServerConfig((url) => ({
          ...routedProviderCfg(url),
          plugin: [pathToFileURL(plugin).href],
        }))
        const prompt = yield* SessionPrompt.Service
        const sessions = yield* Session.Service
        const chat = yield* sessions.create({ title: name })
        yield* llm.push(reply().stop(), reply().text("must not run").stop())

        yield* prompt.prompt({ sessionID: chat.id, agent: "build", parts: [{ type: "text", text: "route" }] })
        const messages = yield* sessions.messages({ sessionID: chat.id })
        const assistants = messages.filter(
          (message): message is SessionV1.WithParts & { info: SessionV1.Assistant } =>
            message.info.role === "assistant",
        )
        expect(assistants).toHaveLength(1)
        expect(JSON.stringify(assistants[0]?.info.error)).toContain(
          "Routed empty-response failure: invalid fallback proposal",
        )
        expect(messages.filter((message) => message.info.routedHandoff)).toHaveLength(0)
        expect(yield* llm.hits).toHaveLength(1)
      }),
    { config: () => routedCfg },
    10_000,
  )
}

it.instance(
  "does not route non-empty terminal text",
  () =>
    Effect.gen(function* () {
      const { directory } = yield* TestInstance
      const plugin = path.join(directory, "protocol-empty-observer.ts")
      const invoked = path.join(directory, "protocol-empty-hook")
      yield* writeText(
        plugin,
        `export default async () => ({ "chat.provider.failure": async (input) => { if (input.failure === "protocol_empty") await Bun.write(${JSON.stringify(invoked)}, "called") } })`,
      )
      const { llm } = yield* useServerConfig((url) => ({
        ...routedProviderCfg(url),
        plugin: [pathToFileURL(plugin).href],
      }))
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Non-empty" })
      yield* llm.text("visible")

      yield* prompt.prompt({ sessionID: chat.id, agent: "build", parts: [{ type: "text", text: "text" }] })
      expect(yield* Effect.promise(() => Bun.file(invoked).exists())).toBe(false)
      expect((yield* sessions.messages({ sessionID: chat.id })).every((message) => !message.info.routedHandoff)).toBe(
        true,
      )
      expect(yield* llm.hits).toHaveLength(1)
    }),
  { config: () => routedCfg },
  10_000,
)

it.instance(
  "does not route valid structured output",
  () =>
    Effect.gen(function* () {
      const { directory } = yield* TestInstance
      const plugin = path.join(directory, "protocol-empty-structured.ts")
      const counter = path.join(directory, "protocol-empty-structured-count")
      yield* writeText(counter, "0")
      yield* writeText(plugin, protocolEmptyCounterPlugin(counter))
      const { llm } = yield* useServerConfig((url) => ({
        ...routedProviderCfg(url),
        plugin: [pathToFileURL(plugin).href],
      }))
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Structured output" })
      yield* llm.push(reply().tool("StructuredOutput", { answer: "structured" }).stop())

      const result = yield* prompt.prompt({
        sessionID: chat.id,
        agent: "build",
        parts: [{ type: "text", text: "return structured data" }],
        format: new SessionV1.OutputFormatJsonSchema({
          type: "json_schema",
          schema: {
            type: "object",
            properties: { answer: { type: "string" } },
            required: ["answer"],
          },
          retryCount: 0,
        }),
      })

      expect(result.info.role).toBe("assistant")
      if (result.info.role === "assistant") {
        expect(result.info.structured).toEqual({ answer: "structured" })
        expect(result.info.error).toBeUndefined()
      }
      expect(yield* protocolEmptyInvocationCount(counter)).toBe(0)
      expect(yield* llm.hits).toHaveLength(1)
    }),
  { config: () => routedCfg },
  10_000,
)

it.instance(
  "does not route a user-consumable file attachment",
  () =>
    Effect.gen(function* () {
      const { directory } = yield* TestInstance
      const plugin = path.join(directory, "protocol-empty-attachment.ts")
      const counter = path.join(directory, "protocol-empty-attachment-count")
      yield* writeText(counter, "0")
      yield* writeText(plugin, protocolEmptyCounterPlugin(counter))
      const { llm } = yield* useServerConfig((url) => ({
        ...routedProviderCfg(url),
        plugin: [pathToFileURL(plugin).href],
      }))
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Attachment response" })
      const gate = defer<void>()
      const attachment = path.join(directory, "attachment.png")
      yield* Effect.promise(() =>
        Bun.write(attachment, Bun.file(path.join(import.meta.dir, "../tool/fixtures/large-image.png"))),
      )
      yield* llm.push(reply().tool("read", { filePath: attachment }).stop())
      yield* llm.hold("The attachment is available.", gate.promise)

      const run = yield* prompt
        .prompt({ sessionID: chat.id, agent: "build", parts: [{ type: "text", text: "read the image" }] })
        .pipe(Effect.forkChild)
      yield* awaitWithTimeout(llm.wait(2), "timed out waiting for attachment continuation", "10 seconds")
      const messages = yield* sessions.messages({ sessionID: chat.id })
      const tool = messages
        .flatMap((message) => message.parts)
        .find(
          (part): part is CompletedToolPart =>
            part.type === "tool" &&
            part.tool === "read" &&
            part.state.status === "completed" &&
            Boolean(
              part.state.attachments?.some(
                (attachment) => attachment.type === "file" && attachment.url.startsWith("data:"),
              ),
            ),
        )

      expect(tool).toBeDefined()
      expect(yield* protocolEmptyInvocationCount(counter)).toBe(0)
      gate.resolve()
      const result = yield* awaitWithTimeout(
        Fiber.join(run),
        "timed out waiting for attachment final response",
        "10 seconds",
      )
      expect(result.parts.some((part) => part.type === "text" && part.text.trim().length > 0)).toBe(true)
      expect(yield* protocolEmptyInvocationCount(counter)).toBe(0)
      expect(yield* llm.hits).toHaveLength(2)
    }),
  { config: () => routedCfg },
  10_000,
)

unix(
  "does not route a tool-only intermediate response before its final response",
  () =>
    Effect.gen(function* () {
      const { directory: dir } = yield* TestInstance
      const llm = yield* TestLLMServer
      const plugin = path.join(dir, "protocol-empty-tool.ts")
      const counter = path.join(dir, "protocol-empty-tool-count")
      const sideEffect = path.join(dir, "protocol-empty-tool-side-effect")
      const gate = defer<void>()
      yield* writeText(counter, "0")
      yield* writeText(plugin, protocolEmptyCounterPlugin(counter))
      yield* writeConfig(dir, { ...routedProviderCfg(llm.url), plugin: [pathToFileURL(plugin).href] })
      const config = yield* Config.Service
      yield* config.invalidate()
      yield* config.get()
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({
        title: "Tool continuation",
        permission: [{ permission: "*", pattern: "*", action: "allow" }],
      })
      yield* llm.tool("bash", {
        command: `printf executed >> ${JSON.stringify(sideEffect)}`,
        workdir: path.resolve(dir),
      })
      yield* llm.hold("final response", gate.promise)

      const run = yield* prompt
        .prompt({ sessionID: chat.id, agent: "build", parts: [{ type: "text", text: "run the tool" }] })
        .pipe(Effect.forkChild)
      yield* awaitWithTimeout(llm.wait(2), "timed out waiting for tool continuation", "10 seconds")
      expect(yield* Effect.promise(() => Bun.file(sideEffect).text())).toBe("executed")
      expect(yield* protocolEmptyInvocationCount(counter)).toBe(0)

      gate.resolve()
      const result = yield* awaitWithTimeout(Fiber.join(run), "timed out waiting for final response", "10 seconds")
      expect(result.parts.some((part) => part.type === "text" && part.text.trim().length > 0)).toBe(true)
      expect(yield* Effect.promise(() => Bun.file(sideEffect).text())).toBe("executed")
      expect(yield* protocolEmptyInvocationCount(counter)).toBe(0)
      expect(yield* llm.hits).toHaveLength(2)
    }),
  { config: () => routedCfg },
  20_000,
)

it.instance(
  "routes a live 429 to one tagged fallback successor",
  () =>
    Effect.gen(function* () {
      const { directory } = yield* TestInstance
      const plugin = path.join(directory, "route-plugin.ts")
      yield* writeText(
        plugin,
        [
          "export default async () => ({",
          '  "chat.provider.failure": (_input, output) => {',
          '    output.action = "fallback"',
          '    output.models.push({ providerID: "fallback", modelID: "fallback-model" })',
          "  },",
          "})",
        ].join("\n"),
      )
      const { llm } = yield* useServerConfig((url) => ({
        ...routedProviderCfg(url),
        plugin: [pathToFileURL(plugin).href],
      }))
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Live handoff" })
      yield* llm.error(429, { error: "provider body must not leak" })
      yield* llm.text("fallback reply")

      const result = yield* awaitWithTimeout(
        prompt.prompt({ sessionID: chat.id, agent: "build", parts: [{ type: "text", text: "route" }] }),
        "timed out waiting for live routed handoff",
        "5 seconds",
      )
      const messages = yield* sessions.messages({ sessionID: chat.id })
      const users = messages.filter((message) => message.info.role === "user")
      const assistants = messages.filter((message) => message.info.role === "assistant")
      const failed = assistants.filter((message) => message.info.routedHandoff)
      const successor = assistants.filter(
        (message): message is SessionV1.WithParts & { info: SessionV1.Assistant } =>
          message.info.role === "assistant" && Boolean(message.info.routedHandoffID),
      )

      expect(users).toHaveLength(1)
      expect(failed).toHaveLength(1)
      expect(successor).toHaveLength(1)
      expect(successor[0]?.info.providerID).toBe(ProviderV2.ID.make("fallback"))
      expect(result.info.id).toBe(successor[0]?.info.id)
      expect(yield* llm.hits).toHaveLength(2)
      expect(JSON.stringify(failed[0]?.info.routedHandoff)).not.toContain("provider body")
    }),
  10_000,
)

it.instance(
  "chains applied handoffs with new IDs and stops exhaustion without native retry",
  () =>
    Effect.gen(function* () {
      const { directory } = yield* TestInstance
      const plugin = path.join(directory, "chained-route-plugin.ts")
      yield* writeText(
        plugin,
        [
          "export default async () => {",
          "  let failures = 0",
          "  return {",
          '    "chat.provider.failure": (input, output) => {',
          '      if (input.failure !== "rate_limit") return',
          "      failures++",
          '      if (failures === 1) { output.action = "fallback"; output.models.push({ providerID: "fallback", modelID: "fallback-model" }) }',
          '      if (failures === 2) { output.action = "fallback"; output.models.push({ providerID: "test", modelID: "test-model" }) }',
          '      if (failures === 3) output.action = "stop"',
          "    },",
          "  }",
          "}",
        ].join("\n"),
      )
      const { llm } = yield* useServerConfig((url) => ({
        ...routedProviderCfg(url),
        plugin: [pathToFileURL(plugin).href],
      }))
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Chained routed failures" })
      yield* llm.error(429, { error: "first" })
      yield* llm.error(429, { error: "second" })
      yield* llm.error(429, { error: "third" })
      yield* llm.error(429, { error: "must not retry" })

      yield* prompt.prompt({ sessionID: chat.id, agent: "build", parts: [{ type: "text", text: "route" }] })

      const assistants = (yield* sessions.messages({ sessionID: chat.id })).filter(
        (message): message is SessionV1.WithParts & { info: SessionV1.Assistant } => message.info.role === "assistant",
      )
      const handoffs = assistants.flatMap((message) => (message.info.routedHandoff ? [message.info.routedHandoff] : []))
      const users = (yield* sessions.messages({ sessionID: chat.id })).filter(
        (message): message is SessionV1.WithParts & { info: SessionV1.User } => message.info.role === "user",
      )
      expect(handoffs).toHaveLength(2)
      expect(handoffs[0]?.id).not.toBe(handoffs[1]?.id)
      expect(handoffs.map((handoff) => handoff.status)).toEqual(["applied", "applied"])
      expect(users).toHaveLength(1)
      expect(assistants.every((assistant) => assistant.info.parentID === users[0]?.info.id)).toBe(true)
      expect(
        assistants.map((assistant) => ({ providerID: assistant.info.providerID, modelID: assistant.info.modelID })),
      ).toEqual([
        { providerID: ProviderV2.ID.make("test"), modelID: ModelV2.ID.make("test-model") },
        { providerID: ProviderV2.ID.make("fallback"), modelID: ModelV2.ID.make("fallback-model") },
        { providerID: ProviderV2.ID.make("test"), modelID: ModelV2.ID.make("test-model") },
      ])
      expect(users[0]?.info.routedHandoff).toMatchObject({ id: handoffs[1]?.id, status: "applied" })
      expect(JSON.stringify(assistants.at(-1)?.info.error)).toContain("Routed provider failure: routing stopped")
      expect(yield* llm.hits).toHaveLength(3)
    }),
  10_000,
)

it.instance(
  "recovers an assistant-only marker with the fallback model",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(routedProviderCfg)
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Marker recovery" })
      const seeded = yield* seedRoutedHandoff(chat.id, { userMarker: false })
      yield* llm.hang

      const fiber = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
      yield* awaitWithTimeout(llm.wait(1), "timed out waiting for marker recovery", "2 seconds")
      const user = yield* MessageV2.get({ sessionID: chat.id, messageID: seeded.user.id })
      const hits = yield* llm.hits
      expect(user.info.role).toBe("user")
      if (user.info.role === "user") expect(user.info.model).toEqual(seeded.handoff.next)
      expect(hits).toHaveLength(1)
      expect(hits[0]?.body.model).toBe("fallback-model")
      yield* Fiber.interrupt(fiber)
    }),
  10_000,
)

it.instance(
  "routes a missing current model through the same user handoff",
  () =>
    Effect.gen(function* () {
      const { directory } = yield* TestInstance
      const plugin = path.join(directory, "config-model-plugin.ts")
      yield* writeText(
        plugin,
        [
          "export default async () => ({",
          '  "chat.provider.failure": (input, output) => {',
          '    if (input.failure !== "config_model") return',
          '    output.action = "fallback"',
          '    output.models.push({ providerID: "fallback", modelID: "fallback-model", variant: "high" })',
          "  },",
          "})",
        ].join("\n"),
      )
      const { llm } = yield* useServerConfig((url) => ({
        ...routedProviderCfg(url),
        provider: {
          ...routedProviderCfg(url).provider,
          fallback: {
            ...routedProviderCfg(url).provider.fallback,
            models: {
              "fallback-model": {
                ...cfg.provider.test.models["test-model"],
                id: "fallback-model",
                reasoning: true,
                variants: { high: {} },
              },
            },
          },
        },
        plugin: [pathToFileURL(plugin).href],
      }))
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Missing configured model" })
      const current = yield* user(chat.id, "route missing model")
      current.model = { providerID: ref.providerID, modelID: ModelV2.ID.make("missing-model") }
      yield* sessions.updateMessage(current)
      yield* sessions.setAgentModel({
        sessionID: chat.id,
        agent: current.agent,
        model: { providerID: current.model.providerID, id: current.model.modelID, variant: "default" },
        time: Date.now(),
      })
      yield* llm.text("fallback reply")

      yield* prompt.loop({ sessionID: chat.id })

      const messages = yield* sessions.messages({ sessionID: chat.id })
      const users = messages.filter((message) => message.info.role === "user")
      const successors = messages.filter(
        (message): message is SessionV1.WithParts & { info: SessionV1.Assistant } =>
          message.info.role === "assistant" && Boolean(message.info.routedHandoffID),
      )
      expect(users).toHaveLength(1)
      expect(successors).toHaveLength(1)
      expect(successors[0]?.info.providerID).toBe(ProviderV2.ID.make("fallback"))
      expect(successors[0]?.info.variant).toBe("high")
      expect((yield* llm.hits).every((hit) => hit.body.model !== "missing-model")).toBe(true)
      expect(users[0]?.info.routedHandoff).toMatchObject({ failure: "config_model", status: "applied" })
    }),
  10_000,
)

it.instance(
  "omits an absent config model fallback variant",
  () =>
    Effect.gen(function* () {
      const { directory } = yield* TestInstance
      const plugin = path.join(directory, "config-model-no-variant-plugin.ts")
      yield* writeText(
        plugin,
        'export default async () => ({ "chat.provider.failure": (input, output) => { if (input.failure === "config_model") { output.action = "fallback"; output.models.push({ providerID: "fallback", modelID: "fallback-model" }) } } })',
      )
      const { llm } = yield* useServerConfig((url) => ({
        ...routedProviderCfg(url),
        plugin: [pathToFileURL(plugin).href],
      }))
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Missing configured model without variant" })
      const current = yield* user(chat.id, "route missing model")
      current.model = { providerID: ref.providerID, modelID: ModelV2.ID.make("missing-model") }
      yield* sessions.updateMessage(current)
      yield* llm.text("fallback reply")

      yield* prompt.loop({ sessionID: chat.id })

      const stored = yield* MessageV2.get({ sessionID: chat.id, messageID: current.id })
      expect(stored.info.role).toBe("user")
      if (stored.info.role === "user") {
        expect(stored.info.model).toEqual({
          providerID: ProviderV2.ID.make("fallback"),
          modelID: ModelV2.ID.make("fallback-model"),
        })
      }
      expect((yield* sessions.get(chat.id)).model?.variant).toBeUndefined()
    }),
  10_000,
)

it.instance(
  "stops an invalid config model fallback variant before persistence or provider work",
  () =>
    Effect.gen(function* () {
      const { directory } = yield* TestInstance
      const plugin = path.join(directory, "invalid-config-model-variant-plugin.ts")
      yield* writeText(
        plugin,
        'export default async () => ({ "chat.provider.failure": (input, output) => { if (input.failure === "config_model") { output.action = "fallback"; output.models.push({ providerID: "fallback", modelID: "fallback-model", variant: "missing" }) } } })',
      )
      const { llm } = yield* useServerConfig((url) => ({
        ...routedProviderCfg(url),
        plugin: [pathToFileURL(plugin).href],
      }))
      const events = yield* EventV2Bridge.Service
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Invalid configured fallback variant" })
      const current = yield* user(chat.id, "missing")
      current.model = { providerID: ref.providerID, modelID: ModelV2.ID.make("missing-model") }
      yield* sessions.updateMessage(current)
      const errors: NonNullable<SessionV1.Assistant["error"]>[] = []
      const off = yield* events.listen((event) => {
        if (event.type !== Session.Event.Error.type) return Effect.void
        const data = event.data as typeof Session.Event.Error.data.Type
        if (data.sessionID === chat.id && data.error) errors.push(data.error)
        return Effect.void
      })

      yield* prompt.loop({ sessionID: chat.id })
      yield* off

      expect(errors).toContainEqual(
        expect.objectContaining({ data: { message: "Routed config/model failure: invalid fallback proposal" } }),
      )
      const stored = yield* MessageV2.get({ sessionID: chat.id, messageID: current.id })
      expect(stored.info.role).toBe("user")
      if (stored.info.role === "user") {
        expect(stored.info.model).toEqual({ providerID: ref.providerID, modelID: ModelV2.ID.make("missing-model") })
        expect(stored.info.routedHandoff).toBeUndefined()
      }
      expect(yield* llm.hits).toHaveLength(0)
      expect(
        (yield* sessions.messages({ sessionID: chat.id })).filter((message) => message.info.role === "assistant"),
      ).toHaveLength(0)
    }),
  10_000,
)

raceNoLLMServer.instance(
  "stops when the config model hook throws without provider work",
  () =>
    Effect.gen(function* () {
      processorModels.length = 0
      yield* Effect.addFinalizer(() =>
        Effect.sync(() => {
          processorModels.length = 0
        }),
      )
      const plugin = yield* Plugin.Service
      const original = plugin.triggerProviderFailure
      ;(plugin as { triggerProviderFailure: typeof plugin.triggerProviderFailure }).triggerProviderFailure = (() =>
        Effect.fail("provider failure hook exploded")) as Plugin.Interface["triggerProviderFailure"]
      yield* Effect.addFinalizer(() =>
        Effect.sync(() => {
          ;(plugin as { triggerProviderFailure: typeof plugin.triggerProviderFailure }).triggerProviderFailure =
            original
        }),
      )
      const events = yield* EventV2Bridge.Service
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Config hook throw" })
      const current = yield* user(chat.id, "missing")
      current.model = { providerID: ref.providerID, modelID: ModelV2.ID.make("missing-model") }
      yield* sessions.updateMessage(current)
      const errors: NonNullable<SessionV1.Assistant["error"]>[] = []
      const off = yield* events.listen((event) => {
        if (event.type !== Session.Event.Error.type) return Effect.void
        const data = event.data as typeof Session.Event.Error.data.Type
        if (data.sessionID === chat.id && data.error) errors.push(data.error)
        return Effect.void
      })

      expect(Exit.isSuccess(yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.exit))).toBe(true)
      yield* off

      expect(errors).toContainEqual(
        expect.objectContaining({ data: { message: "Routed config/model failure: routing stopped" } }),
      )
      expect(processorModels).toHaveLength(0)
    }),
  { config: () => routedNoHttpCfg },
  10_000,
)

it.instance(
  "keeps an unhandled missing model native and unmarked",
  () =>
    Effect.gen(function* () {
      const { directory } = yield* TestInstance
      const plugin = path.join(directory, "unhandled-config-model-plugin.ts")
      const invoked = path.join(directory, "config-model-hook-invoked")
      yield* writeText(
        plugin,
        [
          "export default async () => ({",
          '  "chat.provider.failure": async (input) => {',
          '    if (input.failure === "config_model") await Bun.write(' + JSON.stringify(invoked) + ', "invoked")',
          "  },",
          "})",
        ].join("\n"),
      )
      const { llm: configured } = yield* useServerConfig((url) => ({
        ...providerCfg(url),
        plugin: [pathToFileURL(plugin).href],
      }))
      const events = yield* EventV2Bridge.Service
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Unhandled missing model" })
      const current = yield* user(chat.id, "missing")
      current.model = { providerID: ref.providerID, modelID: ModelV2.ID.make("missing-model") }
      yield* sessions.updateMessage(current)
      const errors: NonNullable<SessionV1.Assistant["error"]>[] = []
      const off = yield* events.listen((event) => {
        if (event.type !== Session.Event.Error.type) return Effect.void
        const data = event.data as typeof Session.Event.Error.data.Type
        if (data.sessionID === chat.id && data.error) errors.push(data.error)
        return Effect.void
      })

      const exit = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.exit)
      yield* off

      expect(Exit.isFailure(exit)).toBe(true)
      if (Exit.isFailure(exit)) expect(ProviderSvc.ModelNotFoundError.isInstance(Cause.squash(exit.cause))).toBe(true)
      expect(yield* Effect.promise(() => Bun.file(invoked).exists())).toBe(true)
      expect(errors).toContainEqual(
        expect.objectContaining({
          data: { message: expect.stringMatching(/^Model not found: test\/missing-model\./) },
        }),
      )
      expect((yield* sessions.messages({ sessionID: chat.id })).every((message) => !message.info.routedHandoff)).toBe(
        true,
      )
      expect(yield* configured.hits).toHaveLength(0)
    }),
  10_000,
)

it.instance(
  "stops handled missing models without assistant or provider work",
  () =>
    Effect.gen(function* () {
      const { directory } = yield* TestInstance
      const plugin = path.join(directory, "stop-config-model-plugin.ts")
      yield* writeText(
        plugin,
        [
          "export default async () => ({",
          '  "chat.provider.failure": (input, output) => {',
          '    if (input.failure !== "config_model") return',
          '    output.action = input.model.modelID === "missing-stop" ? "stop" : "fallback"',
          "  },",
          "})",
        ].join("\n"),
      )
      const { llm } = yield* useServerConfig((url) => ({ ...providerCfg(url), plugin: [pathToFileURL(plugin).href] }))
      const events = yield* EventV2Bridge.Service
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const errors: NonNullable<SessionV1.Assistant["error"]>[] = []
      const off = yield* events.listen((event) => {
        if (event.type !== Session.Event.Error.type) return Effect.void
        const data = event.data as typeof Session.Event.Error.data.Type
        if (data.error) errors.push(data.error)
        return Effect.void
      })

      for (const [modelID, diagnostic] of [
        ["missing-stop", "Routed config/model failure: routing stopped"],
        ["missing-exhausted", "Routed config/model failure: invalid fallback proposal"],
      ]) {
        const chat = yield* sessions.create({ title: modelID })
        const current = yield* user(chat.id, "missing")
        current.model = { providerID: ref.providerID, modelID: ModelV2.ID.make(modelID) }
        yield* sessions.updateMessage(current)
        const exit = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.exit)
        expect(Exit.isSuccess(exit)).toBe(true)
        expect(errors.at(-1)).toMatchObject({ data: { message: diagnostic } })
        expect(JSON.stringify(errors.at(-1))).not.toContain("provider body")
        expect(
          (yield* sessions.messages({ sessionID: chat.id })).filter((message) => message.info.role === "assistant"),
        ).toHaveLength(0)
      }
      yield* off

      expect(yield* llm.hits).toHaveLength(0)
    }),
  10_000,
)

it.instance(
  "stops a missing fallback proposal before later candidates",
  () =>
    Effect.gen(function* () {
      const { directory } = yield* TestInstance
      const plugin = path.join(directory, "invalid-config-model-plugin.ts")
      yield* writeText(
        plugin,
        [
          "export default async () => ({",
          '  "chat.provider.failure": (input, output) => {',
          '    if (input.failure !== "config_model") return',
          '    output.action = "fallback"',
          '    output.models.push({ providerID: "missing", modelID: "missing-model" })',
          '    output.models.push({ providerID: "fallback", modelID: "fallback-model" })',
          "  },",
          "})",
        ].join("\n"),
      )
      const { llm } = yield* useServerConfig((url) => ({
        ...routedProviderCfg(url),
        plugin: [pathToFileURL(plugin).href],
      }))
      const events = yield* EventV2Bridge.Service
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Invalid fallback proposal" })
      const current = yield* user(chat.id, "missing")
      current.model = { providerID: ref.providerID, modelID: ModelV2.ID.make("missing-model") }
      yield* sessions.updateMessage(current)
      const errors: NonNullable<SessionV1.Assistant["error"]>[] = []
      const off = yield* events.listen((event) => {
        if (event.type !== Session.Event.Error.type) return Effect.void
        const data = event.data as typeof Session.Event.Error.data.Type
        if (data.sessionID === chat.id && data.error) errors.push(data.error)
        return Effect.void
      })

      yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.exit)
      yield* off

      expect(errors).toContainEqual(
        expect.objectContaining({ data: { message: "Routed config/model failure: invalid fallback proposal" } }),
      )
      const stored = yield* MessageV2.get({ sessionID: chat.id, messageID: current.id })
      expect(stored.info.role).toBe("user")
      if (stored.info.role === "user") {
        expect(stored.info.model).toEqual({ providerID: ref.providerID, modelID: ModelV2.ID.make("missing-model") })
        expect(stored.info.routedHandoff).toBeUndefined()
      }
      expect(yield* llm.hits).toHaveLength(0)
    }),
  10_000,
)

raceNoLLMServer.instance(
  "selects the first ordered configured fallback with its variant",
  () =>
    Effect.gen(function* () {
      processorCreateStarted.length = 0
      processorModels.length = 0
      yield* Effect.addFinalizer(() =>
        Effect.sync(() => {
          processorCreateStarted.length = 0
          processorModels.length = 0
        }),
      )

      const plugin = yield* Plugin.Service
      const original = plugin.triggerProviderFailure
      ;(plugin as { triggerProviderFailure: typeof plugin.triggerProviderFailure }).triggerProviderFailure = () =>
        Effect.succeed({
          action: "fallback",
          models: [
            { providerID: "fallback", modelID: "fallback-model", variant: "high" },
            { providerID: "fallback", modelID: "second-model" },
          ],
        })
      yield* Effect.addFinalizer(() =>
        Effect.sync(() => {
          ;(plugin as { triggerProviderFailure: typeof plugin.triggerProviderFailure }).triggerProviderFailure =
            original
        }),
      )

      yield* (yield* Config.Service).get()
      const provider = yield* ProviderSvc.Service
      expect(Object.keys((yield* provider.list())[ProviderV2.ID.make("fallback")]?.models ?? {})).toContain(
        "fallback-model",
      )
      expect(
        Exit.isSuccess(
          yield* provider.getModel(ProviderV2.ID.make("fallback"), ModelV2.ID.make("fallback-model")).pipe(Effect.exit),
        ),
      ).toBe(true)
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Ordered fallback" })
      const current = yield* user(chat.id, "missing")
      current.model = { providerID: ref.providerID, modelID: ModelV2.ID.make("missing-model") }
      yield* sessions.updateMessage(current)
      const successorCreated = defer<void>()
      processorCreateStarted.push(successorCreated.resolve)

      const fiber = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
      yield* awaitWithTimeout(
        Effect.promise(() => successorCreated.promise),
        "timed out waiting for ordered fallback successor",
        "2 seconds",
      )

      const stored = yield* MessageV2.get({ sessionID: chat.id, messageID: current.id })
      const session = yield* sessions.get(chat.id)
      const successors = (yield* sessions.messages({ sessionID: chat.id })).filter(
        (message): message is SessionV1.WithParts & { info: SessionV1.Assistant } =>
          message.info.role === "assistant" && Boolean(message.info.routedHandoffID),
      )
      expect(stored.info.role).toBe("user")
      if (stored.info.role === "user") {
        expect(stored.info.model).toEqual({
          providerID: ProviderV2.ID.make("fallback"),
          modelID: ModelV2.ID.make("fallback-model"),
          variant: "high",
        })
      }
      expect(session.model).toEqual({
        providerID: ProviderV2.ID.make("fallback"),
        id: ModelV2.ID.make("fallback-model"),
        variant: "high",
      })
      expect(successors).toHaveLength(1)
      expect(successors[0]?.info.variant).toBe("high")
      expect(processorModels).toHaveLength(1)
      expect(processorModels[0]).toMatchObject({
        providerID: ProviderV2.ID.make("fallback"),
        id: ModelV2.ID.make("fallback-model"),
      })
      yield* Fiber.interrupt(fiber)
    }),
  { config: () => routedNoHttpCfg },
  10_000,
)

raceNoLLMServer.instance(
  "recovers a config model user marker once without requesting its invalid source",
  () =>
    Effect.gen(function* () {
      processorCreateStarted.length = 0
      processorModels.length = 0
      yield* Effect.addFinalizer(() =>
        Effect.sync(() => {
          processorCreateStarted.length = 0
          processorModels.length = 0
        }),
      )

      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Config user marker recovery" })
      const current = (yield* user(chat.id, "recover missing model")) as SessionV1.User
      current.model = { providerID: ref.providerID, modelID: ModelV2.ID.make("missing-model") }
      const handoff = {
        ...routedHandoff(current.id),
        failure: "config_model" as const,
        from: current.model,
      }
      current.routedHandoff = handoff
      yield* sessions.updateMessage(current)
      const successorCreated = defer<void>()
      processorCreateStarted.push(successorCreated.resolve)

      const fiber = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
      yield* awaitWithTimeout(
        Effect.promise(() => successorCreated.promise),
        "timed out waiting for config marker recovery",
        "2 seconds",
      )
      const messages = yield* sessions.messages({ sessionID: chat.id })
      const successors = messages.filter(
        (message): message is SessionV1.WithParts & { info: SessionV1.Assistant } =>
          message.info.role === "assistant" && message.info.routedHandoffID === handoff.id,
      )
      const recovered = yield* MessageV2.get({ sessionID: chat.id, messageID: current.id })
      expect(successors).toHaveLength(1)
      expect(recovered.info.role).toBe("user")
      if (recovered.info.role === "user") expect(recovered.info.model).toEqual(handoff.next)
      expect(successors[0]?.info.routedHandoff).toBeUndefined()
      expect(processorModels).toHaveLength(1)
      expect(processorModels[0]).toMatchObject({
        providerID: ProviderV2.ID.make("fallback"),
        id: ModelV2.ID.make("fallback-model"),
      })
      expect(processorModels.every((model) => model.id !== ModelV2.ID.make("missing-model"))).toBe(true)
      yield* Fiber.interrupt(fiber)
    }),
  { config: () => routedProviderCfg("http://localhost:1/v1") },
  10_000,
)

it.instance(
  "keeps unrelated model resolution defects terminal without invoking the hook",
  () =>
    Effect.gen(function* () {
      const { directory } = yield* TestInstance
      const plugin = path.join(directory, "defect-config-model-plugin.ts")
      const invoked = path.join(directory, "defect-config-model-hook-invoked")
      yield* writeText(
        plugin,
        [
          "export default async () => ({",
          '  "chat.provider.failure": async () => {',
          "    await Bun.write(" + JSON.stringify(invoked) + ', "invoked")',
          "  },",
          "})",
        ].join("\n"),
      )
      const { llm } = yield* useServerConfig((url) => ({ ...providerCfg(url), plugin: [pathToFileURL(plugin).href] }))
      const provider = yield* ProviderSvc.Service
      const original = provider.getModel
      ;(provider as { getModel: typeof provider.getModel }).getModel = () =>
        Effect.die(new Error("unrelated model resolution defect"))
      yield* Effect.addFinalizer(() =>
        Effect.sync(() => {
          ;(provider as { getModel: typeof provider.getModel }).getModel = original
        }),
      )
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Unrelated model resolution defect" })
      yield* user(chat.id, "defect")

      const exit = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.exit)

      expect(Exit.isFailure(exit)).toBe(true)
      if (Exit.isFailure(exit)) expect(String(Cause.squash(exit.cause))).toContain("unrelated model resolution defect")
      expect(yield* Effect.promise(() => Bun.file(invoked).exists())).toBe(false)
      expect((yield* sessions.messages({ sessionID: chat.id })).every((message) => !message.info.routedHandoff)).toBe(
        true,
      )
      expect(yield* llm.hits).toHaveLength(0)
    }),
  10_000,
)

it.instance(
  "preserves native missing model behavior with no plugin",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(providerCfg)
      const events = yield* EventV2Bridge.Service
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Plugin absent missing model" })
      const current = yield* user(chat.id, "missing")
      current.model = { providerID: ref.providerID, modelID: ModelV2.ID.make("missing-model") }
      yield* sessions.updateMessage(current)
      const errors: NonNullable<SessionV1.Assistant["error"]>[] = []
      const off = yield* events.listen((event) => {
        if (event.type !== Session.Event.Error.type) return Effect.void
        const data = event.data as typeof Session.Event.Error.data.Type
        if (data.sessionID === chat.id && data.error) errors.push(data.error)
        return Effect.void
      })

      const exit = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.exit)
      yield* off

      expect(Exit.isFailure(exit)).toBe(true)
      if (Exit.isFailure(exit)) expect(ProviderSvc.ModelNotFoundError.isInstance(Cause.squash(exit.cause))).toBe(true)
      expect(errors).toEqual([
        expect.objectContaining({
          data: { message: expect.stringMatching(/^Model not found: test\/missing-model\./) },
        }),
      ])
      expect((yield* sessions.messages({ sessionID: chat.id })).every((message) => !message.info.routedHandoff)).toBe(
        true,
      )
      expect(yield* llm.hits).toHaveLength(0)
    }),
  10_000,
)

for (const [name, input, message] of [
  ["pending and tagged", { status: "pending", tagged: true }, "routed handoff interrupted"],
  [
    "pending and tagged with stop finish",
    { status: "pending", tagged: true, finish: "stop" },
    "routed handoff interrupted",
  ],
  [
    "pending and tagged with tool-calls finish",
    { status: "pending", tagged: true, finish: "tool-calls" },
    "routed handoff interrupted",
  ],
  [
    "applied tagged tool-calls with open tail",
    { status: "applied", tagged: true, finish: "tool-calls", tail: true },
    "routed handoff interrupted",
  ],
  [
    "applied terminal tagged with open tail",
    { status: "applied", tagged: true, terminal: true, tail: true },
    "routed handoff interrupted",
  ],
  ["applied and tagged nonterminal", { status: "applied", tagged: true }, "routed handoff interrupted"],
  ["applied and tagged cancelled", { status: "applied", tagged: true, cancelled: true }, "MessageAbortedError"],
  ["applied and missing successor", { status: "applied" }, "routed handoff invariant"],
  [
    "applied and mismatched successor",
    { status: "applied", tagged: true, differentTagged: true },
    "routed handoff invariant",
  ],
] as const) {
  noLLMServer.instance(
    `${name} stops without provider work`,
    () =>
      Effect.gen(function* () {
        const prompt = yield* SessionPrompt.Service
        const sessions = yield* Session.Service
        const chat = yield* sessions.create({ title: name })
        const seeded = yield* seedRoutedHandoff(chat.id, input)

        yield* awaitWithTimeout(prompt.loop({ sessionID: chat.id }), `timed out stopping ${name}`, "500 millis")
        const messages = yield* sessions.messages({ sessionID: chat.id })
        const last = messages.findLast(
          (item): item is SessionV1.WithParts & { info: SessionV1.Assistant } => item.info.role === "assistant",
        )
        expect(JSON.stringify(last?.info.error)).toContain(message)
        expect(last?.info.time.completed).toBeDefined()
        expect(seeded.handoff.status).toBe(input.status)
      }),
    { config: routedCfg },
  )
}

noLLMServer.instance(
  "applied terminal handoff exits without a replacement",
  () =>
    Effect.gen(function* () {
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Applied terminal" })
      yield* seedRoutedHandoff(chat.id, { status: "applied", tagged: true, terminal: true })

      const result = yield* awaitWithTimeout(
        prompt.loop({ sessionID: chat.id }),
        "timed out exiting applied terminal",
        "500 millis",
      )
      const messages = yield* sessions.messages({ sessionID: chat.id })
      expect(result.info.role).toBe("assistant")
      if (result.info.role === "assistant") expect(result.info.finish).toBe("stop")
      expect(messages.filter((message) => message.info.role === "assistant")).toHaveLength(2)
    }),
  { config: routedCfg },
)

noLLMServer.instance(
  "pending tagged terminal handoff exits without marking it interrupted",
  () =>
    Effect.gen(function* () {
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Pending terminal" })
      const seeded = yield* seedRoutedHandoff(chat.id, { tagged: true, terminal: true })

      const result = yield* awaitWithTimeout(
        prompt.loop({ sessionID: chat.id }),
        "timed out exiting pending terminal",
        "500 millis",
      )
      const messages = yield* sessions.messages({ sessionID: chat.id })
      const user = messages.find(
        (message): message is SessionV1.WithParts & { info: SessionV1.User } => message.info.role === "user",
      )
      const failed = messages.find(
        (message): message is SessionV1.WithParts & { info: SessionV1.Assistant } =>
          message.info.role === "assistant" && message.info.routedHandoff?.id === seeded.handoff.id,
      )
      const successor = messages.find(
        (message): message is SessionV1.WithParts & { info: SessionV1.Assistant } =>
          message.info.role === "assistant" && message.info.routedHandoffID === seeded.handoff.id,
      )
      if (!user || !failed || !successor) return yield* Effect.die("missing pending terminal handoff messages")
      expect(result.info.id).toBe(successor.info.id)
      expect(successor.info.error).toBeUndefined()
      expect(successor.info.finish).toBe("stop")
      expect(user.info.routedHandoff).toMatchObject({ id: seeded.handoff.id, status: "applied" })
      expect(failed.info.routedHandoff).toMatchObject({ id: seeded.handoff.id, status: "applied" })
      expect(messages.filter((message) => message.info.role === "assistant")).toHaveLength(2)
    }),
  { config: routedCfg },
)

it.instance(
  "authoritative applied handoff suppresses stale markers across work units",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(routedProviderCfg)
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Stale handoff" })
      const seeded = yield* seedRoutedHandoff(chat.id, { status: "applied" })
      seeded.assistant.routedHandoff = { ...seeded.handoff, status: "pending" }
      yield* sessions.updateMessage(seeded.assistant)

      yield* awaitWithTimeout(prompt.loop({ sessionID: chat.id }), "timed out stopping stale handoff", "500 millis")
      expect(yield* llm.calls).toBe(0)

      yield* llm.text("new work")
      yield* user(chat.id, "later work unit")
      yield* prompt.loop({ sessionID: chat.id })

      const messages = yield* sessions.messages({ sessionID: chat.id })
      expect(yield* llm.calls).toBe(1)
      expect(messages.filter((message) => message.info.role === "assistant")).toHaveLength(2)
    }),
  10_000,
)

it.instance(
  "pending handoff bypasses over-budget pre-process compaction",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(routedProviderCfg)
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Pending overflow" })
      const seeded = yield* seedRoutedHandoff(chat.id)
      seeded.assistant.tokens = { ...seeded.assistant.tokens, input: 200_000 }
      yield* sessions.updateMessage(seeded.assistant)
      yield* llm.text("fallback reply")

      yield* prompt.loop({ sessionID: chat.id })

      const messages = yield* sessions.messages({ sessionID: chat.id })
      expect(yield* llm.calls).toBe(1)
      expect(
        messages.filter(
          (message) => message.info.role === "assistant" && message.info.routedHandoffID === seeded.handoff.id,
        ),
      ).toHaveLength(1)
      expect(messages.flatMap((message) => message.parts).some((part) => part.type === "compaction")).toBe(false)
    }),
  10_000,
)

for (const [site, candidates, selected, minContextTokens, variant] of [
  [
    "pre-process",
    [{ providerID: "fallback", modelID: "fallback-large", variant: "high" }],
    "fallback-large",
    101,
    "high",
  ],
  [
    "post-process",
    [{ providerID: "fallback", modelID: "fallback-large", variant: "high" }],
    "fallback-large",
    101,
    "high",
  ],
  [
    "pre-process",
    [
      { providerID: "fallback", modelID: "fallback-bound" },
      { providerID: "fallback", modelID: "fallback-large", variant: "high" },
    ],
    "fallback-large",
    101,
    "high",
  ],
  [
    "post-process",
    [
      { providerID: "fallback", modelID: "fallback-bound" },
      { providerID: "fallback", modelID: "fallback-large", variant: "high" },
    ],
    "fallback-large",
    101,
    "high",
  ],
  ["pre-process", [{ providerID: "fallback", modelID: "fallback-large" }], "fallback-large", 101, undefined],
  ["post-process", [{ providerID: "fallback", modelID: "fallback-large" }], "fallback-large", 101, undefined],
] as const) {
  overflowHarness.instance(
    `${site} overflow selects the ${candidates.length === 1 ? "first" : "second"} capable fallback${variant ? " with an explicit variant" : " without a variant"} without compaction`,
    () =>
      Effect.gen(function* () {
        const { directory } = yield* TestInstance
        const plugin = path.join(directory, `${site}-overflow-plugin.ts`)
        const capture = path.join(directory, `${site}-overflow-input.json`)
        yield* writeText(plugin, overflowPlugin({ capture, action: "fallback", models: candidates }))
        const { llm } = yield* useServerConfig((url) => ({
          ...overflowProviderCfg(url),
          plugin: [pathToFileURL(plugin).href],
        }))
        const events = yield* EventV2Bridge.Service
        const statuses: string[] = []
        const off = yield* events.listen((event) => {
          if (event.type !== SessionStatus.Event.Status.type) return Effect.void
          statuses.push((event.data as typeof SessionStatus.Event.Status.data.Type).status.type)
          return Effect.void
        })
        const prompt = yield* SessionPrompt.Service
        const sessions = yield* Session.Service
        const chat = yield* sessions.create({ title: `${site} overflow fallback` })
        if (site === "pre-process") {
          const seeded = yield* seed(chat.id, { finish: "tool-calls" })
          seeded.assistant.tokens = { ...seeded.assistant.tokens, input: minContextTokens }
          yield* sessions.updateMessage(seeded.assistant)
          yield* llm.text("fallback", { usage: { input: 1, output: 1 } })
          yield* prompt.loop({ sessionID: chat.id })
        } else {
          yield* llm.push(
            reply().text("overflow").usage({ input: minContextTokens, output: 0 }).stop(),
            reply().text("fallback").usage({ input: 1, output: 1 }).stop(),
          )
          yield* prompt.prompt({
            sessionID: chat.id,
            agent: "build",
            parts: [{ type: "text", text: "PROMPT-MUST-NOT-LEAK" }],
          })
        }
        yield* off

        const input = yield* Effect.promise(() => Bun.file(capture).json())
        const messages = yield* sessions.messages({ sessionID: chat.id })
        const users = messages.filter((message) => message.info.role === "user")
        const successors = messages.filter(
          (message): message is SessionV1.WithParts & { info: SessionV1.Assistant } =>
            message.info.role === "assistant" && Boolean(message.info.routedHandoffID),
        )
        expect(input).toMatchObject({ failure: "overflow" })
        expect(JSON.stringify(input)).not.toContain("test-key")
        expect(JSON.stringify(input)).not.toContain("PROMPT-MUST-NOT-LEAK")
        expect(overflowCompactions(chat.id)).toHaveLength(0)
        expect(users).toHaveLength(1)
        expect(successors).toHaveLength(1)
        expect(successors[0]?.info.parentID).toBe(users[0]?.info.id)
        expect(successors[0]?.info.modelID).toBe(ModelV2.ID.make(selected))
        expect(successors[0]?.info.variant).toBe(variant)
        expect(users[0]?.info.routedHandoff?.next).toEqual({
          providerID: ProviderV2.ID.make("fallback"),
          modelID: ModelV2.ID.make(selected),
          ...(variant ? { variant } : {}),
        })
        expect((yield* sessions.get(chat.id)).model?.variant).toBe(variant)
        if (site === "post-process") {
          const completed = messages.find(
            (message): message is SessionV1.WithParts & { info: SessionV1.Assistant } =>
              message.info.role === "assistant" && message.info.routedHandoffID === undefined,
          )
          expect(completed?.info.finish).toBe("stop")
          expect(completed?.info.error).toBeUndefined()
          expect(completed?.info.routedHandoff).toBeUndefined()
          expect(completed?.parts).toContainEqual(expect.objectContaining({ type: "text", text: "overflow" }))
          expect(JSON.stringify((yield* llm.hits)[1]?.body.messages)).toContain("overflow")
        }
        expect(statuses.slice(0, -1)).not.toContain("idle")
      }),
    { config: () => overflowProviderCfg("http://localhost:1/v1") },
    10_000,
  )
}

overflowHarness.instance(
  "post-process provider overflow marks the failed assistant and omits it from the fallback transcript",
  () =>
    Effect.gen(function* () {
      const { directory } = yield* TestInstance
      const plugin = path.join(directory, "post-process-provider-overflow-plugin.ts")
      const capture = path.join(directory, "post-process-provider-overflow-input.json")
      yield* writeText(
        plugin,
        overflowPlugin({
          capture,
          action: "fallback",
          models: [{ providerID: "fallback", modelID: "fallback-large" }],
        }),
      )
      const { llm } = yield* useServerConfig((url) => ({
        ...overflowProviderCfg(url),
        plugin: [pathToFileURL(plugin).href],
      }))
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Post-process provider overflow" })
      yield* llm.error(413, { error: { message: "request entity too large" } })
      yield* llm.text("fallback")

      yield* prompt.prompt({ sessionID: chat.id, agent: "build", parts: [{ type: "text", text: "provider overflow" }] })

      const assistants = (yield* sessions.messages({ sessionID: chat.id })).filter(
        (message): message is SessionV1.WithParts & { info: SessionV1.Assistant } => message.info.role === "assistant",
      )
      const failed = assistants.find((message) => message.info.routedHandoff?.failure === "overflow")
      expect(overflowCompactions(chat.id)).toHaveLength(0)
      const input = yield* Effect.promise(() => Bun.file(capture).json())
      expect(input).toMatchObject({ failure: "overflow" })
      expect(failed?.info.finish).toBe("error")
      expect(failed?.info.error).toBeDefined()
      expect(failed?.parts.some((part) => part.type === "text")).toBe(false)
      expect(failed?.info.routedHandoff).toMatchObject({ status: "applied", failure: "overflow" })
      expect(
        assistants.filter((message) => message.info.routedHandoffID === failed?.info.routedHandoff?.id),
      ).toHaveLength(1)
      expect((yield* sessions.get(chat.id)).model?.variant).toBeUndefined()
      expect(
        ((yield* llm.hits)[1]?.body.messages as Array<{ role: string }>).filter(
          (message) => message.role === "assistant",
        ),
      ).toHaveLength(0)
    }),
  { config: () => overflowProviderCfg("http://localhost:1/v1") },
  10_000,
)

overflowHarness.instance(
  "provider overflow rejects a candidate at the current usable bound and compacts once",
  () =>
    Effect.gen(function* () {
      const { directory } = yield* TestInstance
      const plugin = path.join(directory, "provider-overflow-bound-plugin.ts")
      yield* writeText(
        plugin,
        overflowPlugin({
          capture: path.join(directory, "provider-overflow-bound-input.json"),
          action: "fallback",
          models: [{ providerID: "fallback", modelID: "fallback-bound" }],
        }),
      )
      const { llm } = yield* useServerConfig((url) => ({
        ...overflowProviderCfg(url),
        plugin: [pathToFileURL(plugin).href],
      }))
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Provider overflow bound" })
      yield* llm.error(413, { error: { message: "request entity too large" } })

      yield* prompt
        .prompt({ sessionID: chat.id, agent: "build", parts: [{ type: "text", text: "provider overflow" }] })
        .pipe(Effect.exit)

      expect(overflowCompactions(chat.id)).toHaveLength(1)
      expect(overflowCompactions(chat.id)[0]?.overflow).toBe(true)
      const failed = (yield* sessions.messages({ sessionID: chat.id })).find(
        (message): message is SessionV1.WithParts & { info: SessionV1.Assistant } => message.info.role === "assistant",
      )
      expect(failed?.info.finish).toBe("error")
      expect(failed?.info.error).toBeDefined()
    }),
  { config: () => overflowProviderCfg("http://localhost:1/v1") },
  10_000,
)

overflowHarness.instance(
  "provider overflow with unavailable usable lower bound compacts instead of routing with a zero minimum",
  () =>
    Effect.gen(function* () {
      const { directory } = yield* TestInstance
      const plugin = path.join(directory, "unavailable-estimate-overflow-plugin.ts")
      yield* writeText(
        plugin,
        overflowPlugin({
          capture: path.join(directory, "unavailable-estimate-overflow-input.json"),
          action: "fallback",
          models: [{ providerID: "fallback", modelID: "fallback-large" }],
        }),
      )
      yield* useServerConfig((url) => ({
        ...overflowProviderCfg(url),
        provider: {
          ...overflowProviderCfg(url).provider,
          test: {
            ...overflowProviderCfg(url).provider.test,
            models: {
              "test-model": {
                ...overflowProviderCfg(url).provider.test.models["test-model"],
                limit: { context: 0, output: 20 },
              },
            },
          },
        },
        plugin: [pathToFileURL(plugin).href],
      }))
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Unavailable request estimate" })
      const seeded = yield* seed(chat.id, { finish: "tool-calls" })
      seeded.assistant.tokens = { ...seeded.assistant.tokens, input: 100 }
      yield* sessions.updateMessage(seeded.assistant)

      yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.exit)

      expect(overflowCompactions(chat.id)).toHaveLength(1)
      expect(overflowCompactions(chat.id)[0]?.overflow).toBe(false)
      expect((yield* sessions.messages({ sessionID: chat.id })).every((message) => !message.info.routedHandoff)).toBe(
        true,
      )
    }),
  { config: () => overflowProviderCfg("http://localhost:1/v1") },
  10_000,
)

cappedOverflowHarness.instance(
  "provider overflow active output cap rejects the bound candidate and selects the larger fallback",
  () =>
    Effect.gen(function* () {
      const { directory } = yield* TestInstance
      const plugin = path.join(directory, "capped-provider-overflow-plugin.ts")
      const capture = path.join(directory, "capped-provider-overflow-input.json")
      yield* writeText(
        plugin,
        overflowPlugin({
          capture,
          action: "fallback",
          models: [
            { providerID: "fallback", modelID: "fallback-bound" },
            { providerID: "fallback", modelID: "fallback-large" },
          ],
        }),
      )
      const { llm } = yield* useServerConfig((url) => ({
        ...overflowProviderCfg(url),
        plugin: [pathToFileURL(plugin).href],
      }))
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Capped provider overflow" })
      yield* llm.error(413, { error: { message: "request entity too large" } })
      yield* llm.text("fallback")

      yield* prompt.prompt({ sessionID: chat.id, agent: "build", parts: [{ type: "text", text: "provider overflow" }] })

      const successors = (yield* sessions.messages({ sessionID: chat.id })).filter(
        (message): message is SessionV1.WithParts & { info: SessionV1.Assistant } =>
          message.info.role === "assistant" && Boolean(message.info.routedHandoffID),
      )
      const input = yield* Effect.promise(() => Bun.file(capture).json())
      expect(input).toMatchObject({ failure: "overflow" })
      expect(overflowCompactions(chat.id)).toHaveLength(0)
      expect(successors).toHaveLength(1)
      expect(successors[0]?.info.modelID).toBe(ModelV2.ID.make("fallback-large"))
    }),
  { config: () => overflowProviderCfg("http://localhost:1/v1") },
  10_000,
)

overflowHarness.instance(
  "pre-process overflow stops when the provider failure hook throws",
  () =>
    Effect.gen(function* () {
      const plugin = yield* Plugin.Service
      const original = plugin.triggerProviderFailure
      ;(plugin as { triggerProviderFailure: typeof plugin.triggerProviderFailure }).triggerProviderFailure = (input) =>
        input.failure === "overflow" ? Effect.fail("provider failure hook exploded") : original(input)
      yield* Effect.addFinalizer(() =>
        Effect.sync(() => {
          ;(plugin as { triggerProviderFailure: typeof plugin.triggerProviderFailure }).triggerProviderFailure =
            original
        }),
      )
      const events = yield* EventV2Bridge.Service
      const errors: NonNullable<SessionV1.Assistant["error"]>[] = []
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Overflow hook throw" })
      const off = yield* events.listen((event) => {
        if (event.type !== Session.Event.Error.type) return Effect.void
        const data = event.data as typeof Session.Event.Error.data.Type
        if (data.sessionID === chat.id && data.error) errors.push(data.error)
        return Effect.void
      })
      const seeded = yield* seed(chat.id, { finish: "tool-calls" })
      seeded.assistant.tokens = { ...seeded.assistant.tokens, input: 101 }
      yield* sessions.updateMessage(seeded.assistant)

      yield* prompt.loop({ sessionID: chat.id })
      yield* off

      expect(errors).toContainEqual(
        expect.objectContaining({ data: { message: "Routed overflow failure: routing stopped" } }),
      )
      expect(overflowCompactions(chat.id)).toHaveLength(0)
    }),
  { config: () => overflowProviderCfg("http://localhost:1/v1") },
  10_000,
)

for (const [site, kind, action, models, expectedCompactions] of [
  ["pre-process", "all incapable", "fallback", [{ providerID: "fallback", modelID: "fallback-small" }], 1],
  ["post-process", "all incapable", "fallback", [{ providerID: "fallback", modelID: "fallback-small" }], 1],
  ["pre-process", "unhandled", undefined, undefined, 1],
  ["post-process", "unhandled", undefined, undefined, 1],
  ["pre-process", "handled stop", "stop", [], 0],
  ["post-process", "handled stop", "stop", [], 0],
  ["pre-process", "no candidates", "fallback", [], 1],
  ["post-process", "no candidates", "fallback", [], 1],
] as const) {
  overflowHarness.instance(
    `${site} overflow ${kind} ${
      expectedCompactions === 1 ? "creates native compaction once" : "stops without compaction"
    }`,
    () =>
      Effect.gen(function* () {
        const { directory } = yield* TestInstance
        const plugin = path.join(directory, `${site}-native-overflow-plugin.ts`)
        const capture = path.join(directory, `${site}-native-overflow-input.json`)
        if (action) {
          yield* writeText(plugin, overflowPlugin({ capture, action, models }))
        }
        yield* writeConfig(directory, {
          ...overflowProviderCfg((yield* TestLLMServer).url),
          ...(action ? { plugin: [pathToFileURL(plugin).href] } : {}),
        })
        const service = yield* Config.Service
        yield* service.invalidate()
        yield* service.get()
        const prompt = yield* SessionPrompt.Service
        const sessions = yield* Session.Service
        const chat = yield* sessions.create({ title: `${site} native overflow` })
        const events = yield* EventV2Bridge.Service
        const errors: NonNullable<SessionV1.Assistant["error"]>[] = []
        const off = yield* events.listen((event) => {
          if (event.type !== Session.Event.Error.type) return Effect.void
          const data = event.data as typeof Session.Event.Error.data.Type
          if (data.sessionID === chat.id && data.error) errors.push(data.error)
          return Effect.void
        })
        if (site === "pre-process") {
          const seeded = yield* seed(chat.id, { finish: "tool-calls" })
          seeded.assistant.tokens = { ...seeded.assistant.tokens, input: 100 }
          yield* sessions.updateMessage(seeded.assistant)
          yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.exit)
        } else {
          const llm = yield* TestLLMServer
          yield* llm.push(reply().text("overflow").usage({ input: 100, output: 0 }).stop())
          yield* prompt
            .prompt({ sessionID: chat.id, agent: "build", parts: [{ type: "text", text: "native" }] })
            .pipe(Effect.exit)
        }
        yield* off
        expect(overflowCompactions(chat.id)).toHaveLength(expectedCompactions)
        if (expectedCompactions) expect(overflowCompactions(chat.id)[0]?.overflow).toBe(false)
        if (action) expect(yield* Effect.promise(() => Bun.file(capture).exists())).toBe(true)
        if (kind === "handled stop") {
          expect(errors).toContainEqual(
            expect.objectContaining({ data: { message: "Routed overflow failure: routing stopped" } }),
          )
          expect(JSON.stringify(errors)).not.toContain("test-key")
          if (site === "post-process") {
            const failed = (yield* sessions.messages({ sessionID: chat.id })).find(
              (message): message is SessionV1.WithParts & { info: SessionV1.Assistant } =>
                message.info.role === "assistant",
            )
            expect(failed?.info.error).toMatchObject({ data: { message: "Routed overflow failure: routing stopped" } })
          }
        }
      }),
    { config: () => overflowProviderCfg("http://localhost:1/v1") },
    10_000,
  )
}

for (const [site, models] of [
  ["pre-process", [{ providerID: "test", modelID: "test-model" }]],
  ["post-process", [{ providerID: "test", modelID: "test-model" }]],
  [
    "pre-process",
    [
      { providerID: "fallback", modelID: "fallback-large" },
      { providerID: "fallback", modelID: "fallback-large" },
    ],
  ],
  [
    "post-process",
    [
      { providerID: "fallback", modelID: "fallback-large" },
      { providerID: "fallback", modelID: "fallback-large" },
    ],
  ],
  ["pre-process", [{ providerID: "missing", modelID: "missing" }]],
  ["post-process", [{ providerID: "missing", modelID: "missing" }]],
  ["pre-process", [{ providerID: "fallback", modelID: "fallback-large", variant: "missing" }]],
  ["post-process", [{ providerID: "fallback", modelID: "fallback-large", variant: "missing" }]],
] as const) {
  overflowHarness.instance(
    `${site} overflow invalid proposal stops without compaction`,
    () =>
      Effect.gen(function* () {
        const { directory } = yield* TestInstance
        const plugin = path.join(directory, `${site}-invalid-overflow-plugin.ts`)
        const capture = path.join(directory, `${site}-invalid-overflow-input.json`)
        yield* writeText(plugin, overflowPlugin({ capture, action: "fallback", models }))
        const { llm } = yield* useServerConfig((url) => ({
          ...overflowProviderCfg(url),
          plugin: [pathToFileURL(plugin).href],
        }))
        const prompt = yield* SessionPrompt.Service
        const sessions = yield* Session.Service
        const chat = yield* sessions.create({ title: `${site} invalid overflow` })
        if (site === "pre-process") {
          const seeded = yield* seed(chat.id, { finish: "tool-calls" })
          seeded.assistant.tokens = { ...seeded.assistant.tokens, input: 100 }
          yield* sessions.updateMessage(seeded.assistant)
          yield* prompt.loop({ sessionID: chat.id })
        } else {
          yield* llm.push(reply().text("overflow").usage({ input: 100, output: 0 }).stop())
          yield* prompt.prompt({ sessionID: chat.id, agent: "build", parts: [{ type: "text", text: "invalid" }] })
        }
        const messages = yield* sessions.messages({ sessionID: chat.id })
        expect(yield* Effect.promise(() => Bun.file(capture).exists())).toBe(true)
        expect(overflowCompactions(chat.id)).toHaveLength(0)
        expect(messages.filter((message) => message.info.routedHandoff)).toHaveLength(0)
        if (site === "post-process") {
          const completed = messages.find(
            (message): message is SessionV1.WithParts & { info: SessionV1.Assistant } =>
              message.info.role === "assistant",
          )
          expect(completed?.info.finish).toBe("stop")
          expect(completed?.info.error).toBeUndefined()
          expect(completed?.parts).toContainEqual(expect.objectContaining({ type: "text", text: "overflow" }))
        }
      }),
    { config: () => overflowProviderCfg("http://localhost:1/v1") },
    10_000,
  )
}

it.instance(
  "pending handoff defers a queued subtask until its tagged successor finishes once",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(routedProviderCfg)
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Pending subtask" })
      const seeded = yield* seedRoutedHandoff(chat.id)
      yield* addSubtask(chat.id, seeded.user.id)
      yield* llm.tool("glob", { pattern: "*" })
      yield* llm.text("fallback completed")
      yield* llm.text("subtask reply")

      yield* prompt.loop({ sessionID: chat.id })

      const messages = yield* sessions.messages({ sessionID: chat.id })
      const tagged = messages.filter(
        (message) => message.info.role === "assistant" && message.info.routedHandoffID === seeded.handoff.id,
      )
      const subtasks = messages.filter(
        (message) => message.info.role === "assistant" && message.info.agent === "general",
      )
      expect(yield* llm.calls).toBe(3)
      expect(tagged).toHaveLength(1)
      expect(subtasks).toHaveLength(1)
      expect(messages.indexOf(tagged[0]!)).toBeLessThan(messages.indexOf(subtasks[0]!))
    }),
  10_000,
)

for (const [name, userMarker] of [
  ["processor marker before user apply", false],
  ["user pending before successor create", true],
] as const) {
  noLLMServer.instance(
    `cancel clears ${name} without recovery activity`,
    () =>
      Effect.gen(function* () {
        const prompt = yield* SessionPrompt.Service
        const sessions = yield* Session.Service
        const chat = yield* sessions.create({ title: `Cancel ${name}` })
        const seeded = yield* seedRoutedHandoff(chat.id, { userMarker })

        yield* prompt.cancel(chat.id)

        const messages = yield* sessions.messages({ sessionID: chat.id })
        expect(messages.filter((message) => message.info.routedHandoff?.status === "pending")).toHaveLength(0)
        expect(
          messages.filter(
            (message) => message.info.role === "assistant" && message.info.routedHandoffID === seeded.handoff.id,
          ),
        ).toHaveLength(0)
        expect(messages.filter((message) => message.info.role === "assistant")).toHaveLength(1)
      }),
    { config: routedCfg },
  )
}

it.instance("loop exits without an LLM request for interrupted orphan tool calls", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig(providerCfg)
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const chat = yield* sessions.create({ title: "Pinned" })
    const seeded = yield* seed(chat.id, { finish: "stop" })
    yield* sessions.updatePart({
      id: PartID.ascending(),
      messageID: seeded.assistant.id,
      sessionID: chat.id,
      type: "tool",
      callID: "interrupted-call",
      tool: "edit",
      state: {
        status: "error",
        input: {},
        error: "Tool execution aborted",
        metadata: { interrupted: true },
        time: { start: 1, end: 2 },
      },
    })

    const result = yield* prompt.loop({ sessionID: chat.id })
    expect(result.info.id).toBe(seeded.assistant.id)
    expect(yield* llm.hits).toHaveLength(0)
  }),
)

it.instance("loop calls LLM and returns assistant message", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig(providerCfg)
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const chat = yield* sessions.create({
      title: "Pinned",
      permission: [{ permission: "*", pattern: "*", action: "allow" }],
    })
    yield* prompt.prompt({
      sessionID: chat.id,
      agent: "build",
      noReply: true,
      parts: [{ type: "text", text: "hello" }],
    })
    yield* llm.text("world")

    const result = yield* prompt.loop({ sessionID: chat.id })
    expect(result.info.role).toBe("assistant")
    const parts = result.parts.filter((p) => p.type === "text")
    expect(parts.some((p) => p.type === "text" && p.text === "world")).toBe(true)
    expect(yield* llm.hits).toHaveLength(1)
  }),
)

withMcpInstructions.instance(
  "loop includes MCP instructions in model system context",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(providerCfg)
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({
        title: "Pinned",
        permission: [{ permission: "*", pattern: "*", action: "allow" }],
      })
      yield* llm.hang
      yield* user(chat.id, "hello")

      const fiber = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
      yield* awaitWithTimeout(llm.wait(1), "timed out waiting for MCP instruction request", "10 seconds")

      const hits = yield* llm.hits
      const body = JSON.stringify(hits[0]?.body)
      expect(body).toContain('<server name=\\"guide-server\\">')
      expect(body).toContain("Use lookup before mutate.")
      yield* Fiber.interrupt(fiber)
    }),
  15_000,
)

it.instance("legacy prompt emits message events without session.next events", () =>
  Effect.gen(function* () {
    const events = yield* EventV2Bridge.Service
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const chat = yield* sessions.create({
      title: "Pinned",
      agent: "plan",
      model: { providerID: ProviderV2.ID.make("old"), id: ModelV2.ID.make("old-model") },
    })
    const seen: string[] = []
    const off = yield* events.listen((event) => {
      seen.push(event.type)
      return Effect.void
    })

    const first = yield* prompt.prompt({
      sessionID: chat.id,
      agent: "build",
      model: ref,
      noReply: true,
      parts: [{ type: "text", text: "hello" }],
    })
    const second = yield* prompt.prompt({
      sessionID: chat.id,
      agent: "build",
      noReply: true,
      parts: [{ type: "text", text: "again" }],
    })
    yield* off

    expect(first.info.role).toBe("user")
    expect(second.info.role).toBe("user")
    if (first.info.role === "user" && second.info.role === "user") {
      expect(first.info.model).toEqual(ref)
      expect(second.info.model).toEqual(ref)
    }
    expect(yield* sessions.get(chat.id)).toMatchObject({
      agent: "build",
      model: { providerID: ref.providerID, id: ref.modelID },
    })
    expect(seen).toContain(Session.Event.Updated.type)
    expect(seen).toContain(MessageV2.Event.Updated.type)
    expect(seen).toContain(MessageV2.Event.PartUpdated.type)
    expect(seen.filter((type) => type.startsWith("session.next."))).toEqual([])
  }),
)

it.instance("loop surfaces content-filter finishes as session errors", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig(providerCfg)
    const events = yield* EventV2Bridge.Service
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const chat = yield* sessions.create({ title: "Pinned" })
    const errors: NonNullable<SessionV1.Assistant["error"]>[] = []
    const expected = {
      name: "ContentFilterError",
      data: { message: "The response was blocked by the provider's content filter" },
    } satisfies NonNullable<SessionV1.Assistant["error"]>
    const off = yield* events.listen((event) => {
      if (event.type !== Session.Event.Error.type) return Effect.void
      const data = event.data as typeof Session.Event.Error.data.Type
      if (data.sessionID === chat.id && data.error) errors.push(data.error)
      return Effect.void
    })

    yield* prompt.prompt({
      sessionID: chat.id,
      agent: "build",
      noReply: true,
      parts: [{ type: "text", text: "hello" }],
    })
    yield* llm.push(reply().text("partial response").contentFilter())

    const result = yield* prompt.loop({ sessionID: chat.id })
    const stored = yield* MessageV2.get({ sessionID: chat.id, messageID: result.info.id })
    yield* off

    expect(yield* llm.hits).toHaveLength(1)
    expect(result.info.role).toBe("assistant")
    expect(stored.info.role).toBe("assistant")
    if (result.info.role === "assistant" && stored.info.role === "assistant") {
      expect(result.info.finish).toBe("content-filter")
      expect(result.info.error).toEqual(expected)
      expect(stored.info.error).toEqual(result.info.error)
      expect(errors).toContainEqual(expected)
    }
    expect(result.parts).toEqual(
      expect.arrayContaining([expect.objectContaining({ type: "text", text: "partial response" })]),
    )
  }),
)

it.instance("loop stops provider overflow instead of auto-compacting when disabled", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig((url) => ({
      ...providerCfg(url),
      compaction: { auto: false },
    }))
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const chat = yield* sessions.create({ title: "Pinned" })

    yield* llm.error(413, { error: { message: "request entity too large" } })
    yield* prompt.prompt({
      sessionID: chat.id,
      agent: "build",
      noReply: true,
      parts: [{ type: "text", text: "hello" }],
    })

    const result = yield* prompt.loop({ sessionID: chat.id })
    const messages = yield* sessions.messages({ sessionID: chat.id })

    expect(result.info.role).toBe("assistant")
    if (result.info.role === "assistant") {
      expect(result.info.error?.name).toBe("ContextOverflowError")
      expect(result.info.finish).toBe("error")
    }
    expect(messages.some((message) => message.parts.some((part) => part.type === "compaction"))).toBe(false)
  }),
)

noLLMServer.instance.skip(
  "prompt emits v2 prompted and synthetic events (v2 projector disabled)",
  () =>
    Effect.gen(function* () {
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Pinned" })

      yield* prompt.prompt({
        sessionID: chat.id,
        agent: "build",
        noReply: true,
        parts: [
          { type: "text", text: "hello v2" },
          {
            type: "file",
            mime: "text/plain",
            filename: "note.txt",
            url: "data:text/plain;base64,bm90ZSBjb250ZW50",
          },
        ],
      })

      const messages = yield* SessionV2.Service.use((session) => session.messages({ sessionID: chat.id })).pipe(
        Effect.provide(
          LayerNode.compile(SessionV2.node, [
            [SessionExecution.node, SessionExecution.noopLayer],
            [LocationServiceMap.node, locationServiceMapLayer],
          ]),
        ),
      )
      const { db } = yield* Database.Service
      const row = yield* db
        .select()
        .from(SessionMessageTable)
        .where(eq(SessionMessageTable.session_id, chat.id))
        .get()
        .pipe(Effect.orDie)
      expect(messages.find((message) => message.type === "user")).toMatchObject({ type: "user", text: "hello v2" })
      expect(typeof row?.data.time.created).toBe("number")
      expect(messages).toEqual(
        expect.arrayContaining([
          expect.objectContaining({ type: "synthetic", text: expect.stringContaining("Called the Read tool") }),
          expect.objectContaining({ type: "synthetic", text: "note content" }),
        ]),
      )
    }),
  { config: cfg },
)

it.instance("static loop returns assistant text through local provider", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig(providerCfg)
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const session = yield* sessions.create({
      title: "Prompt provider",
      permission: [{ permission: "*", pattern: "*", action: "allow" }],
    })

    yield* prompt.prompt({
      sessionID: session.id,
      agent: "build",
      noReply: true,
      parts: [{ type: "text", text: "hello" }],
    })

    yield* llm.text("world")

    const result = yield* prompt.loop({ sessionID: session.id })
    expect(result.info.role).toBe("assistant")
    expect(result.parts.some((part) => part.type === "text" && part.text === "world")).toBe(true)
    expect(yield* llm.hits).toHaveLength(1)
    expect(yield* llm.pending).toBe(0)
  }),
)

it.instance("static loop consumes queued replies across turns", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig(providerCfg)
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const session = yield* sessions.create({
      title: "Prompt provider turns",
      permission: [{ permission: "*", pattern: "*", action: "allow" }],
    })

    yield* prompt.prompt({
      sessionID: session.id,
      agent: "build",
      noReply: true,
      parts: [{ type: "text", text: "hello one" }],
    })

    yield* llm.text("world one")

    const first = yield* prompt.loop({ sessionID: session.id })
    expect(first.info.role).toBe("assistant")
    expect(first.parts.some((part) => part.type === "text" && part.text === "world one")).toBe(true)

    yield* prompt.prompt({
      sessionID: session.id,
      agent: "build",
      noReply: true,
      parts: [{ type: "text", text: "hello two" }],
    })

    yield* llm.text("world two")

    const second = yield* prompt.loop({ sessionID: session.id })
    expect(second.info.role).toBe("assistant")
    expect(second.parts.some((part) => part.type === "text" && part.text === "world two")).toBe(true)

    expect(yield* llm.hits).toHaveLength(2)
    expect(yield* llm.pending).toBe(0)
  }),
)

it.instance("loop continues when finish is tool-calls", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig(providerCfg)
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const session = yield* sessions.create({
      title: "Pinned",
      permission: [{ permission: "*", pattern: "*", action: "allow" }],
    })
    yield* prompt.prompt({
      sessionID: session.id,
      agent: "build",
      noReply: true,
      parts: [{ type: "text", text: "hello" }],
    })
    yield* llm.tool("first", { value: "first" })
    yield* llm.text("second")

    const result = yield* prompt.loop({ sessionID: session.id })
    expect(yield* llm.calls).toBe(2)
    expect(result.info.role).toBe("assistant")
    if (result.info.role === "assistant") {
      expect(result.parts.some((part) => part.type === "text" && part.text === "second")).toBe(true)
      expect(result.info.finish).toBe("stop")
    }
  }),
)

it.instance("glob tool keeps instance context during prompt runs", () =>
  Effect.gen(function* () {
    const { dir, llm } = yield* useServerConfig(providerCfg)
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const session = yield* sessions.create({
      title: "Glob context",
      permission: [{ permission: "*", pattern: "*", action: "allow" }],
    })
    const file = path.join(dir, "probe.txt")
    yield* writeText(file, "probe")

    yield* prompt.prompt({
      sessionID: session.id,
      agent: "build",
      noReply: true,
      parts: [{ type: "text", text: "find text files" }],
    })
    yield* llm.tool("glob", { pattern: "**/*.txt" })
    yield* llm.text("done")

    const result = yield* prompt.loop({ sessionID: session.id })
    expect(result.info.role).toBe("assistant")

    const msgs = yield* MessageV2.filterCompactedEffect(session.id)
    const tool = msgs
      .flatMap((msg) => msg.parts)
      .find(
        (part): part is CompletedToolPart =>
          part.type === "tool" && part.tool === "glob" && part.state.status === "completed",
      )
    if (!tool) return

    expect(tool.state.output).toContain(file)
    expect(tool.state.output).not.toContain("No context found for instance")
    expect(result.parts.some((part) => part.type === "text" && part.text === "done")).toBe(true)
  }),
)

it.instance("loop continues when finish is stop but assistant has tool parts", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig(providerCfg)
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const session = yield* sessions.create({
      title: "Pinned",
      permission: [{ permission: "*", pattern: "*", action: "allow" }],
    })
    yield* prompt.prompt({
      sessionID: session.id,
      agent: "build",
      noReply: true,
      parts: [{ type: "text", text: "hello" }],
    })
    yield* llm.push(reply().tool("first", { value: "first" }).stop())
    yield* llm.text("second")

    const result = yield* prompt.loop({ sessionID: session.id })
    expect(yield* llm.calls).toBe(2)
    expect(result.info.role).toBe("assistant")
    if (result.info.role === "assistant") {
      expect(result.parts.some((part) => part.type === "text" && part.text === "second")).toBe(true)
      expect(result.info.finish).toBe("stop")
    }
  }),
)

it.instance("failed subtask preserves metadata on error tool state", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig((url) => ({
      ...providerCfg(url),
      agent: {
        general: {
          model: "test/missing-model",
        },
      },
    }))
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const chat = yield* sessions.create({ title: "Pinned" })
    yield* llm.tool("task", {
      description: "inspect bug",
      prompt: "look into the cache key path",
      subagent_type: "general",
    })
    yield* llm.text("done")
    const msg = yield* user(chat.id, "hello")
    yield* addSubtask(chat.id, msg.id)

    const result = yield* prompt.loop({ sessionID: chat.id })
    expect(result.info.role).toBe("assistant")
    expect(yield* llm.calls).toBe(2)

    const msgs = yield* MessageV2.filterCompactedEffect(chat.id)
    const taskMsg = msgs.find((item) => item.info.role === "assistant" && item.info.agent === "general")
    expect(taskMsg?.info.role).toBe("assistant")
    if (!taskMsg || taskMsg.info.role !== "assistant") return

    const tool = errorTool(taskMsg.parts)
    if (!tool) return

    expect(tool.state.error).toContain("Tool execution failed")
    expect(tool.state.metadata).toBeDefined()
    expect(tool.state.metadata?.sessionId).toBeDefined()
    expect(tool.state.metadata?.model).toEqual({
      providerID: ProviderV2.ID.make("test"),
      modelID: ModelV2.ID.make("missing-model"),
    })
  }),
)

it.instance("subtask child inherits parent session external_directory allow", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig(providerCfg)
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const chat = yield* sessions.create({
      title: "Parent",
      permission: [{ permission: "external_directory", pattern: "/tmp/allowed/*", action: "allow" }],
    })
    yield* llm.text("done")
    const msg = yield* user(chat.id, "hello")
    yield* addSubtask(chat.id, msg.id)

    yield* prompt.loop({ sessionID: chat.id })

    const kids = yield* sessions.children(chat.id)
    expect(kids).toHaveLength(1)
    const child = kids[0]!
    const rules = child.permission ?? []
    expect(rules).toEqual(
      expect.arrayContaining([{ permission: "external_directory", pattern: "/tmp/allowed/*", action: "allow" }]),
    )
    expect(Permission.evaluate("external_directory", "/tmp/allowed/file", rules).action).toBe("allow")
    expect(Permission.evaluate("task", "anything", rules).action).toBe("deny")
  }),
)

noLLMServer.instance("prompt tools replace previous prompt tool rules", () =>
  Effect.gen(function* () {
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const session = yield* sessions.create({ title: "Prompt tools" })

    yield* prompt.prompt({
      sessionID: session.id,
      agent: "build",
      noReply: true,
      tools: { bash: false },
      parts: [{ type: "text", text: "first" }],
    })
    yield* prompt.prompt({
      sessionID: session.id,
      agent: "build",
      noReply: true,
      tools: { read: true },
      parts: [{ type: "text", text: "second" }],
    })

    const reloaded = yield* sessions.get(session.id)
    expect(reloaded.permission).toEqual([{ permission: "read", pattern: "*", action: "allow" }])
    expect(Permission.evaluate("bash", "anything", reloaded.permission ?? []).action).toBe("ask")
  }),
)

it.instance(
  "running subtask preserves metadata after tool-call transition",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(providerCfg)
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Pinned" })
      yield* llm.hang
      const msg = yield* user(chat.id, "hello")
      yield* addSubtask(chat.id, msg.id)

      const fiber = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)

      const tool = yield* pollWithTimeout(
        Effect.gen(function* () {
          const msgs = yield* MessageV2.filterCompactedEffect(chat.id)
          const taskMsg = msgs.find((item) => item.info.role === "assistant" && item.info.agent === "general")
          const tool = taskMsg?.parts.find((part): part is SessionV1.ToolPart => part.type === "tool")
          if (tool?.state.status === "running" && tool.state.metadata?.sessionId) return tool
        }),
        "timed out waiting for running subtask metadata",
      )

      if (tool.state.status !== "running") return
      expect(typeof tool.state.metadata?.sessionId).toBe("string")
      expect(tool.state.title).toBeDefined()
      expect(tool.state.metadata?.model).toBeDefined()

      yield* prompt.cancel(chat.id)
      yield* Fiber.await(fiber)
    }),
  5_000,
)

it.instance(
  "running task tool preserves metadata after tool-call transition",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(providerCfg)
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({
        title: "Pinned",
        permission: [{ permission: "*", pattern: "*", action: "allow" }],
      })
      yield* llm.tool("task", {
        description: "inspect bug",
        prompt: "look into the cache key path",
        subagent_type: "general",
      })
      yield* llm.hang
      yield* user(chat.id, "hello")

      const fiber = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)

      const tool = yield* pollWithTimeout(
        Effect.gen(function* () {
          const msgs = yield* MessageV2.filterCompactedEffect(chat.id)
          const assistant = msgs.findLast((item) => item.info.role === "assistant" && item.info.agent === "build")
          const tool = assistant?.parts.find(
            (part): part is SessionV1.ToolPart => part.type === "tool" && part.tool === "task",
          )
          if (tool?.state.status === "running" && tool.state.metadata?.sessionId) return tool
        }),
        "timed out waiting for running task metadata",
      )

      if (tool.state.status !== "running") return
      expect(typeof tool.state.metadata?.sessionId).toBe("string")
      expect(tool.state.title).toBe("inspect bug")
      expect(tool.state.metadata?.model).toBeDefined()

      yield* prompt.cancel(chat.id)
      yield* Fiber.await(fiber)
    }),
  10_000,
)

it.instance(
  "loop sets status to busy then idle",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(providerCfg)
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const status = yield* SessionStatus.Service

      yield* llm.hang

      const chat = yield* sessions.create({})
      yield* user(chat.id, "hi")

      const fiber = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
      yield* llm.wait(1)
      expect((yield* status.get(chat.id)).type).toBe("busy")
      yield* prompt.cancel(chat.id)
      yield* Fiber.await(fiber)
      expect((yield* status.get(chat.id)).type).toBe("idle")
    }),
  3_000,
)

// Cancel semantics

it.instance("cancel interrupts loop and resolves with an assistant message", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig(providerCfg)
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const chat = yield* sessions.create({ title: "Pinned" })
    yield* seed(chat.id)

    yield* llm.hang

    yield* user(chat.id, "more")

    const fiber = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
    yield* llm.wait(1)
    yield* waitForBusy(chat.id)
    yield* prompt.cancel(chat.id)
    const exit = yield* Fiber.await(fiber)
    expect(Exit.isSuccess(exit)).toBe(true)
    if (Exit.isSuccess(exit)) {
      expect(exit.value.info.role).toBe("assistant")
    }
  }),
)

it.instance("cancel records MessageAbortedError on interrupted process", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig(providerCfg)
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const chat = yield* sessions.create({ title: "Pinned" })
    yield* llm.hang
    yield* user(chat.id, "hello")

    const fiber = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
    yield* llm.wait(1)
    yield* waitForBusy(chat.id)
    yield* prompt.cancel(chat.id)
    const exit = yield* Fiber.await(fiber)
    expect(Exit.isSuccess(exit)).toBe(true)
    if (Exit.isSuccess(exit)) {
      const info = exit.value.info
      if (info.role === "assistant") {
        expect(info.error?.name).toBe("MessageAbortedError")
      }
    }
  }),
)

raceNoLLMServer.instance(
  "finalizes assistant when cancelled before processor creation completes",
  () =>
    Effect.gen(function* () {
      processorCreateStarted.length = 0
      yield* Effect.addFinalizer(() =>
        Effect.sync(() => {
          processorCreateStarted.length = 0
        }),
      )

      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Processor creation race" })

      yield* prompt.prompt({
        sessionID: chat.id,
        agent: "build",
        noReply: true,
        parts: [{ type: "text", text: "first" }],
      })

      const firstCreate = defer<void>()
      processorCreateStarted.push(firstCreate.resolve)
      const first = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
      yield* Effect.promise(() => firstCreate.promise)

      yield* prompt.cancel(chat.id)
      const firstExit = yield* Fiber.await(first)
      expect(Exit.isSuccess(firstExit)).toBe(true)

      let messages = yield* sessions.messages({ sessionID: chat.id })
      const firstInterrupted = messages.at(-1)
      expect(firstInterrupted?.info.role).toBe("assistant")
      expect(firstInterrupted?.parts).toHaveLength(0)
      if (firstInterrupted?.info.role === "assistant") {
        expect(firstInterrupted.info.finish).toBeUndefined()
        expect(firstInterrupted.info.time.completed).toBeNumber()
        expect(firstInterrupted.info.error?.name).toBe("MessageAbortedError")
      }

      yield* prompt.prompt({
        sessionID: chat.id,
        agent: "build",
        noReply: true,
        parts: [{ type: "text", text: "second" }],
      })

      const secondCreate = defer<void>()
      processorCreateStarted.push(secondCreate.resolve)
      const second = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
      yield* Effect.promise(() => secondCreate.promise)

      yield* prompt.cancel(chat.id)
      const secondExit = yield* Fiber.await(second)
      expect(Exit.isSuccess(secondExit)).toBe(true)

      messages = yield* sessions.messages({ sessionID: chat.id })
      const poisonMessages = messages.filter(
        (message) =>
          message.info.role === "assistant" &&
          message.parts.length === 0 &&
          !message.info.finish &&
          !message.info.time.completed &&
          !message.info.error,
      )
      expect(poisonMessages).toHaveLength(0)

      const interruptedMessages = messages.filter(
        (message) =>
          message.info.role === "assistant" &&
          message.parts.length === 0 &&
          message.info.time.completed &&
          message.info.error?.name === "MessageAbortedError",
      )
      expect(interruptedMessages).toHaveLength(2)

      const lastUser = messages.at(-2)
      const lastAssistant = messages.at(-1)
      expect(lastUser?.info.role).toBe("user")
      expect(lastAssistant?.info.role).toBe("assistant")
      if (lastUser?.info.role === "user" && lastAssistant?.info.role === "assistant") {
        expect(lastAssistant.info.parentID).toBe(lastUser?.info.id)
      }
    }),
  { config: cfg },
  3_000,
)

noLLMServer.instance(
  "cancel finalizes subtask tool state",
  () =>
    Effect.gen(function* () {
      const ready = yield* Deferred.make<void>()
      const aborted = yield* Deferred.make<void>()
      const registry = yield* ToolRegistry.Service
      const { task } = yield* registry.named()
      const original = task.execute
      task.execute = (_args, ctx) =>
        Effect.callback<never>((_resume) => {
          ctx.abort.addEventListener("abort", () => succeedVoid(aborted), { once: true })
          if (ctx.abort.aborted) succeedVoid(aborted)
          succeedVoid(ready)
          return Effect.sync(() => succeedVoid(aborted))
        })
      yield* Effect.addFinalizer(() => Effect.sync(() => void (task.execute = original)))

      const { prompt, chat } = yield* boot()
      const msg = yield* user(chat.id, "hello")
      yield* addSubtask(chat.id, msg.id)

      const fiber = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
      yield* awaitWithTimeout(Deferred.await(ready), "timed out waiting for task tool to start", "10 seconds")
      yield* prompt.cancel(chat.id)

      const exit = yield* Fiber.await(fiber)
      expect(Exit.isSuccess(exit)).toBe(true)
      yield* awaitWithTimeout(Deferred.await(aborted), "timed out waiting for task tool abort", "10 seconds")

      const msgs = yield* MessageV2.filterCompactedEffect(chat.id)
      const taskMsg = msgs.find((item) => item.info.role === "assistant" && item.info.agent === "general")
      expect(taskMsg?.info.role).toBe("assistant")
      if (!taskMsg || taskMsg.info.role !== "assistant") return

      const tool = toolPart(taskMsg.parts)
      expect(tool?.type).toBe("tool")
      if (!tool) return

      expect(tool.state.status).not.toBe("running")
      expect(taskMsg.info.time.completed).toBeDefined()
      expect(taskMsg.info.finish).toBeDefined()
    }),
  { config: cfg },
  30_000,
)

it.instance(
  "cancel propagates from slash command subtask to child session",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(providerCfg)
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const status = yield* SessionStatus.Service
      const chat = yield* sessions.create({ title: "Pinned" })
      yield* llm.hang
      const msg = yield* user(chat.id, "hello")
      yield* addSubtask(chat.id, msg.id)

      const fiber = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
      yield* llm.wait(1)

      const msgs = yield* MessageV2.filterCompactedEffect(chat.id)
      const taskMsg = msgs.find((item) => item.info.role === "assistant" && item.info.agent === "general")
      const tool = taskMsg ? toolPart(taskMsg.parts) : undefined
      const sessionID = tool?.state.status === "running" ? tool.state.metadata?.sessionId : undefined
      expect(typeof sessionID).toBe("string")
      if (typeof sessionID !== "string") throw new Error("missing child session id")
      const childID = SessionID.make(sessionID)
      expect((yield* status.get(childID)).type).toBe("busy")

      yield* prompt.cancel(chat.id)
      const exit = yield* Fiber.await(fiber)
      expect(Exit.isSuccess(exit)).toBe(true)

      expect((yield* status.get(chat.id)).type).toBe("idle")
      expect((yield* status.get(childID)).type).toBe("idle")
    }),
  10_000,
)

it.instance(
  "cancel with queued callers resolves all cleanly",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(providerCfg)
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Pinned" })
      yield* llm.hang
      yield* user(chat.id, "hello")

      const a = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
      yield* llm.wait(1)
      const b = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
      yield* Effect.sleep(50)

      yield* prompt.cancel(chat.id)
      const [exitA, exitB] = yield* Effect.all([Fiber.await(a), Fiber.await(b)])
      expect(Exit.isSuccess(exitA)).toBe(true)
      expect(Exit.isSuccess(exitB)).toBe(true)
      if (Exit.isSuccess(exitA) && Exit.isSuccess(exitB)) {
        expect(exitA.value.info.id).toBe(exitB.value.info.id)
      }
    }),
  { git: true },
  10_000,
)

// Queue semantics

noLLMServer.instance("concurrent loop callers get same result", () =>
  Effect.gen(function* () {
    const { prompt, run, chat } = yield* boot()
    yield* seed(chat.id, { finish: "stop" })

    const [a, b] = yield* Effect.all([prompt.loop({ sessionID: chat.id }), prompt.loop({ sessionID: chat.id })], {
      concurrency: "unbounded",
    })

    expect(a.info.id).toBe(b.info.id)
    expect(a.info.role).toBe("assistant")
    yield* run.assertNotBusy(chat.id)
  }),
)

it.instance("concurrent loop callers all receive same error result", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig(providerCfg)
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const chat = yield* sessions.create({ title: "Pinned" })

    yield* llm.fail("boom")
    yield* user(chat.id, "hello")

    const [a, b] = yield* Effect.all([prompt.loop({ sessionID: chat.id }), prompt.loop({ sessionID: chat.id })], {
      concurrency: "unbounded",
    })
    expect(a.info.id).toBe(b.info.id)
    expect(a.info.role).toBe("assistant")
  }),
)

it.instance("prompt submitted during an active run is included in the next LLM input", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig(providerCfg)
    const gate = yield* Deferred.make<void>()
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const chat = yield* sessions.create({ title: "Pinned" })

    yield* llm.hold("first", deferredAsPromise(gate))
    yield* llm.text("second")

    const a = yield* prompt
      .prompt({
        sessionID: chat.id,
        agent: "build",
        model: ref,
        parts: [{ type: "text", text: "first" }],
      })
      .pipe(Effect.forkChild)

    yield* llm.wait(1)
    yield* waitForBusy(chat.id)

    const id = MessageID.ascending()
    const b = yield* prompt
      .prompt({
        sessionID: chat.id,
        messageID: id,
        agent: "build",
        model: ref,
        parts: [{ type: "text", text: "second" }],
      })
      .pipe(Effect.forkChild)

    yield* pollWithTimeout(
      sessions
        .messages({ sessionID: chat.id })
        .pipe(
          Effect.map((msgs) => (msgs.some((msg) => msg.info.role === "user" && msg.info.id === id) ? true : undefined)),
        ),
      "timed out waiting for second prompt to save",
    )

    yield* Deferred.succeed(gate, void 0)

    const [ea, eb] = yield* Effect.all([Fiber.await(a), Fiber.await(b)])
    expect(Exit.isSuccess(ea)).toBe(true)
    expect(Exit.isSuccess(eb)).toBe(true)
    expect(yield* llm.calls).toBe(2)

    const msgs = yield* sessions.messages({ sessionID: chat.id })
    const assistants = msgs.filter((msg) => msg.info.role === "assistant")
    expect(assistants).toHaveLength(2)
    const last = assistants.at(-1)
    if (!last || last.info.role !== "assistant") throw new Error("expected second assistant")
    expect(last.info.parentID).toBe(id)
    expect(last.parts.some((part) => part.type === "text" && part.text === "second")).toBe(true)

    const inputs = yield* llm.inputs
    expect(inputs).toHaveLength(2)
    const messages = inputs.at(-1)?.messages
    if (!Array.isArray(messages)) throw new Error("expected LLM messages")
    expect(messages.at(-1)).toEqual({ role: "user", content: "second" })
  }),
)

it.instance("assertNotBusy fails with BusyError when loop running", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig(providerCfg)
    const prompt = yield* SessionPrompt.Service
    const run = yield* SessionRunState.Service
    const sessions = yield* Session.Service
    yield* llm.hang

    const chat = yield* sessions.create({})
    yield* user(chat.id, "hi")

    const fiber = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
    yield* llm.wait(1)
    yield* waitForBusy(chat.id)

    const exit = yield* run.assertNotBusy(chat.id).pipe(Effect.exit)
    expect(Exit.isFailure(exit)).toBe(true)
    if (Exit.isFailure(exit)) {
      expect(Cause.squash(exit.cause)).toBeInstanceOf(Session.BusyError)
      expect(Cause.squash(exit.cause)).toMatchObject({ _tag: "SessionBusyError", sessionID: chat.id })
    }

    yield* prompt.cancel(chat.id)
    yield* Fiber.await(fiber)
  }),
)

noLLMServer.instance("assertNotBusy succeeds when idle", () =>
  Effect.gen(function* () {
    const run = yield* SessionRunState.Service
    const sessions = yield* Session.Service

    const chat = yield* sessions.create({})
    const exit = yield* run.assertNotBusy(chat.id).pipe(Effect.exit)
    expect(Exit.isSuccess(exit)).toBe(true)
  }),
)

// Shell semantics

it.instance("shell rejects with BusyError when loop running", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig(providerCfg)
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const chat = yield* sessions.create({ title: "Pinned" })
    yield* llm.hang
    yield* user(chat.id, "hi")

    const fiber = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
    yield* llm.wait(1)
    yield* waitForBusy(chat.id)

    const exit = yield* prompt.shell({ sessionID: chat.id, agent: "build", command: "echo hi" }).pipe(Effect.exit)
    expect(Exit.isFailure(exit)).toBe(true)
    if (Exit.isFailure(exit)) {
      expect(Cause.squash(exit.cause)).toBeInstanceOf(Session.BusyError)
      expect(Cause.squash(exit.cause)).toMatchObject({ _tag: "SessionBusyError", sessionID: chat.id })
    }

    yield* prompt.cancel(chat.id)
    yield* Fiber.await(fiber)
  }),
)

unixNoLLMServer(
  "shell captures stdout and stderr in completed tool output",
  () =>
    Effect.gen(function* () {
      const { prompt, run, chat } = yield* boot()
      const result = yield* prompt.shell({
        sessionID: chat.id,
        agent: "build",
        command: "printf out && printf err >&2",
      })

      expect(result.info.role).toBe("assistant")
      const tool = completedTool(result.parts)
      if (!tool) return

      expect(tool.state.output).toContain("out")
      expect(tool.state.output).toContain("err")
      expect(tool.state.metadata.output).toContain("out")
      expect(tool.state.metadata.output).toContain("err")
      yield* run.assertNotBusy(chat.id)
    }),
  { config: cfg },
)

unixNoLLMServer(
  "shell completes a fast command on the preferred shell",
  () =>
    Effect.gen(function* () {
      const { directory: dir } = yield* TestInstance
      const { prompt, run, chat } = yield* boot()
      const result = yield* prompt.shell({
        sessionID: chat.id,
        agent: "build",
        command: "pwd",
      })

      expect(result.info.role).toBe("assistant")
      const tool = completedTool(result.parts)
      if (!tool) return

      expect(tool.state.input.command).toBe("pwd")
      expect(tool.state.output).toContain(dir)
      expect(tool.state.metadata.output).toContain(dir)
      yield* run.assertNotBusy(chat.id)
    }),
  { config: cfg },
)

unixNoLLMServer(
  "shell uses configured shell over env shell",
  () =>
    withSh(() =>
      Effect.gen(function* () {
        if (!(yield* hasBash)) return

        const { prompt, chat } = yield* boot()
        const result = yield* prompt.shell({
          sessionID: chat.id,
          agent: "build",
          command: "[[ 1 -eq 1 ]] && printf configured",
        })

        const tool = completedTool(result.parts)
        if (!tool) return
        expect(tool.state.output).toContain("configured")
      }),
    ),
  { config: { ...cfg, shell: "bash" } },
  30_000,
)

unixNoLLMServer(
  "shell commands can change directory after startup",
  () =>
    withSh(() =>
      Effect.gen(function* () {
        const { directory: dir } = yield* TestInstance
        const { prompt, run, chat } = yield* boot()
        const parent = path.dirname(dir)
        const result = yield* prompt.shell({
          sessionID: chat.id,
          agent: "build",
          command: "cd .. && pwd",
        })

        expect(result.info.role).toBe("assistant")
        const tool = completedTool(result.parts)
        if (!tool) return

        expect(tool.state.output).toContain(parent)
        expect(tool.state.metadata.output).toContain(parent)
        yield* run.assertNotBusy(chat.id)
      }),
    ),
  { config: cfg },
)

unixNoLLMServer(
  "shell lists files from the project directory",
  () =>
    Effect.gen(function* () {
      const { directory: dir } = yield* TestInstance
      const { prompt, run, chat } = yield* boot()
      yield* writeText(path.join(dir, "README.md"), "# e2e\n")

      const result = yield* prompt.shell({
        sessionID: chat.id,
        agent: "build",
        command: "command ls",
      })

      expect(result.info.role).toBe("assistant")
      const tool = completedTool(result.parts)
      if (!tool) return

      expect(tool.state.input.command).toBe("command ls")
      expect(tool.state.output).toContain("README.md")
      expect(tool.state.metadata.output).toContain("README.md")
      yield* run.assertNotBusy(chat.id)
    }),
  { config: cfg },
)

unixNoLLMServer(
  "shell captures stderr from a failing command",
  () =>
    Effect.gen(function* () {
      const { prompt, run, chat } = yield* boot()
      const result = yield* prompt.shell({
        sessionID: chat.id,
        agent: "build",
        command: "command -v __nonexistent_cmd_e2e__ || echo 'not found' >&2; exit 1",
      })

      expect(result.info.role).toBe("assistant")
      const tool = completedTool(result.parts)
      if (!tool) return

      expect(tool.state.output).toContain("not found")
      expect(tool.state.metadata.output).toContain("not found")
      yield* run.assertNotBusy(chat.id)
    }),
  { config: cfg },
)

unixNoLLMServer(
  "shell updates running metadata before process exit",
  () =>
    withSh(() =>
      Effect.gen(function* () {
        const { prompt, chat } = yield* boot()

        const fiber = yield* prompt
          .shell({ sessionID: chat.id, agent: "build", command: "printf first && sleep 0.2 && printf second" })
          .pipe(Effect.forkChild)

        yield* pollWithTimeout(
          Effect.gen(function* () {
            const msgs = yield* MessageV2.filterCompactedEffect(chat.id)
            const taskMsg = msgs.find((item) => item.info.role === "assistant")
            const tool = taskMsg ? toolPart(taskMsg.parts) : undefined
            if (tool?.state.status === "running" && tool.state.metadata?.output.includes("first")) return true
          }),
          "timed out waiting for running shell metadata",
        )

        const exit = yield* Fiber.await(fiber)
        expect(Exit.isSuccess(exit)).toBe(true)
      }),
    ),
  { config: cfg },
  30_000,
)

it.instance(
  "loop waits while shell runs and starts after shell exits",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(providerCfg)
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({
        title: "Pinned",
        permission: [{ permission: "*", pattern: "*", action: "allow" }],
      })
      yield* llm.text("after-shell")

      const sh = yield* prompt
        .shell({ sessionID: chat.id, agent: "build", command: "sleep 0.2" })
        .pipe(Effect.forkChild)
      yield* waitForBusy(chat.id)

      const loop = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
      yield* Effect.sleep(50)

      expect(yield* llm.calls).toBe(0)

      yield* Fiber.await(sh)
      const exit = yield* Fiber.await(loop)

      expect(Exit.isSuccess(exit)).toBe(true)
      if (Exit.isSuccess(exit)) {
        expect(exit.value.info.role).toBe("assistant")
        expect(exit.value.parts.some((part) => part.type === "text" && part.text === "after-shell")).toBe(true)
      }
      expect(yield* llm.calls).toBe(1)
    }),
  { git: true },
  10_000,
)

it.instance(
  "shell completion resumes queued loop callers",
  () =>
    Effect.gen(function* () {
      const { llm } = yield* useServerConfig(providerCfg)
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({
        title: "Pinned",
        permission: [{ permission: "*", pattern: "*", action: "allow" }],
      })
      yield* llm.text("done")

      const sh = yield* prompt
        .shell({ sessionID: chat.id, agent: "build", command: "sleep 0.2" })
        .pipe(Effect.forkChild)
      yield* waitForBusy(chat.id)

      const a = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
      const b = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
      yield* Effect.sleep(50)

      expect(yield* llm.calls).toBe(0)

      yield* Fiber.await(sh)
      const [ea, eb] = yield* Effect.all([Fiber.await(a), Fiber.await(b)])

      expect(Exit.isSuccess(ea)).toBe(true)
      expect(Exit.isSuccess(eb)).toBe(true)
      if (Exit.isSuccess(ea) && Exit.isSuccess(eb)) {
        expect(ea.value.info.id).toBe(eb.value.info.id)
        expect(ea.value.info.role).toBe("assistant")
      }
      expect(yield* llm.calls).toBe(1)
    }),
  { git: true },
  10_000,
)

unix(
  "command ! expansion uses configured shell over env shell",
  () =>
    withSh(() =>
      Effect.gen(function* () {
        if (!(yield* hasBash)) return
        const { llm } = yield* useServerConfig((url) => ({
          ...providerCfg(url),
          shell: "bash",
          command: {
            probe: {
              template: "Probe: !`[[ 1 -eq 1 ]] && printf configured`",
            },
          },
        }))

        const { prompt, chat } = yield* boot()
        yield* llm.text("done")

        const result = yield* prompt.command({
          sessionID: chat.id,
          command: "probe",
          arguments: "",
        })

        expect(result.info.role).toBe("assistant")
        const inputs = yield* llm.inputs
        expect(JSON.stringify(inputs.at(-1)?.messages)).toContain("configured")
      }),
    ),
  30_000,
)

unixNoLLMServer(
  "cancel interrupts shell and resolves cleanly",
  () =>
    withSh(() =>
      Effect.gen(function* () {
        const { prompt, run, chat } = yield* boot()
        const { directory: dir } = yield* TestInstance
        const afs = yield* FSUtil.Service
        const ready = path.join(dir, ".shell-ready")

        const sh = yield* prompt
          .shell({ sessionID: chat.id, agent: "build", command: ": > '.shell-ready'; sleep 30" })
          .pipe(Effect.forkChild)
        yield* pollWithTimeout(
          afs.existsSafe(ready).pipe(Effect.map((exists) => (exists ? (true as const) : undefined))),
          "shell never created readiness marker",
        )

        yield* prompt.cancel(chat.id)

        const status = yield* SessionStatus.Service
        expect((yield* status.get(chat.id)).type).toBe("idle")
        const busy = yield* run.assertNotBusy(chat.id).pipe(Effect.exit)
        expect(Exit.isSuccess(busy)).toBe(true)

        const exit = yield* Fiber.await(sh)
        expect(Exit.isSuccess(exit)).toBe(true)
        if (Exit.isSuccess(exit)) {
          expect(exit.value.info.role).toBe("assistant")
          const tool = completedTool(exit.value.parts)
          if (tool) {
            expect(tool.state.output).toContain("User aborted the command")
          }
        }
      }),
    ),
  { git: true, config: cfg },
  30_000,
)

unixNoLLMServer(
  "cancel persists aborted shell result when shell ignores TERM",
  () =>
    withSh(() =>
      Effect.gen(function* () {
        const { prompt, chat } = yield* boot()
        const { directory: dir } = yield* TestInstance
        const afs = yield* FSUtil.Service
        const ready = path.join(dir, ".trap-ready")

        const sh = yield* prompt
          .shell({
            sessionID: chat.id,
            agent: "build",
            // Touch marker AFTER trap installs so the test waits for the actual
            // ignore-TERM state before cancelling; otherwise SIGTERM can arrive
            // before `trap` runs and the escalation path is never exercised.
            command: `trap '' TERM; touch "${ready}"; sleep 30`,
          })
          .pipe(Effect.forkChild)

        yield* Effect.gen(function* () {
          while (!(yield* afs.existsSafe(ready))) {
            yield* Effect.sleep(Duration.millis(10))
          }
        }).pipe(Effect.timeout(Duration.seconds(5)))

        yield* prompt.cancel(chat.id)

        const exit = yield* Fiber.await(sh)
        expect(Exit.isSuccess(exit)).toBe(true)
        if (Exit.isSuccess(exit)) {
          expect(exit.value.info.role).toBe("assistant")
          const tool = completedTool(exit.value.parts)
          if (tool) {
            expect(tool.state.output).toContain("User aborted the command")
          }
        }
      }),
    ),
  { git: true, config: cfg },
  30_000,
)

unix(
  "cancel finalizes interrupted bash tool output through normal truncation",
  () =>
    Effect.gen(function* () {
      const { dir, llm } = yield* useServerConfig(providerCfg)
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({
        title: "Interrupted bash truncation",
        permission: [{ permission: "*", pattern: "*", action: "allow" }],
      })

      yield* prompt.prompt({
        sessionID: chat.id,
        agent: "build",
        noReply: true,
        parts: [{ type: "text", text: "run bash" }],
      })

      yield* llm.tool("bash", {
        command:
          'i=0; while [ "$i" -lt 4000 ]; do printf "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx %05d\\n" "$i"; i=$((i + 1)); done; printf truncation-ready; sleep 30',
        timeout: 30_000,
        workdir: path.resolve(dir),
      })

      const run = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
      yield* llm.wait(1)
      yield* pollWithTimeout(
        Effect.gen(function* () {
          const msgs = yield* MessageV2.filterCompactedEffect(chat.id)
          const assistant = msgs.findLast((item) => item.info.role === "assistant")
          const tool = assistant ? toolPart(assistant.parts) : undefined
          if (tool?.state.status === "running" && tool.state.metadata?.output.includes("truncation-ready")) return true
        }),
        "timed out waiting for truncated shell output",
      )
      yield* prompt.cancel(chat.id)

      const exit = yield* Fiber.await(run)
      expect(Exit.isSuccess(exit)).toBe(true)
      if (Exit.isFailure(exit)) return

      const tool = completedTool(exit.value.parts)
      if (!tool) return

      expect(tool.state.metadata.truncated).toBe(true)
      expect(typeof tool.state.metadata.outputPath).toBe("string")
      expect(tool.state.output).toMatch(/\.\.\.output truncated\.\.\./)
      expect(tool.state.output).toMatch(/Full output saved to:\s+\S+/)
      expect(tool.state.output).not.toContain("Tool execution aborted")
    }),
  { git: true },
  30_000,
)

unixNoLLMServer(
  "cancel interrupts loop queued behind shell",
  () =>
    Effect.gen(function* () {
      const { prompt, chat } = yield* boot()

      const sh = yield* prompt.shell({ sessionID: chat.id, agent: "build", command: "sleep 30" }).pipe(Effect.forkChild)
      yield* waitForBusy(chat.id)

      const loop = yield* prompt.loop({ sessionID: chat.id }).pipe(Effect.forkChild)
      yield* Effect.sleep(50)

      yield* prompt.cancel(chat.id)

      const exit = yield* Fiber.await(loop)
      expect(Exit.isSuccess(exit)).toBe(true)
      if (Exit.isSuccess(exit)) {
        const tool = completedTool(exit.value.parts)
        expect(tool?.state.output).toContain("User aborted the command")
      }

      yield* Fiber.await(sh)
    }),
  { git: true, config: cfg },
  30_000,
)

unixNoLLMServer(
  "shell rejects when another shell is already running",
  () =>
    withSh(() =>
      Effect.gen(function* () {
        const { prompt, chat } = yield* boot()

        const a = yield* prompt
          .shell({ sessionID: chat.id, agent: "build", command: "sleep 30" })
          .pipe(Effect.forkChild)
        yield* waitForBusy(chat.id)

        const exit = yield* prompt.shell({ sessionID: chat.id, agent: "build", command: "echo hi" }).pipe(Effect.exit)
        expect(Exit.isFailure(exit)).toBe(true)
        if (Exit.isFailure(exit)) {
          expect(Cause.squash(exit.cause)).toBeInstanceOf(Session.BusyError)
        }

        yield* prompt.cancel(chat.id)
        yield* Fiber.await(a)
      }),
    ),
  { git: true, config: cfg },
  30_000,
)

// Abort signal propagation tests for inline tool execution

function hangUntilAborted(tool: { execute: (...args: any[]) => any }) {
  return Effect.gen(function* () {
    const ready = yield* Deferred.make<void>()
    const aborted = yield* Deferred.make<void>()
    const original = tool.execute
    tool.execute = (_args: any, ctx: any) => {
      ctx.abort.addEventListener("abort", () => succeedVoid(aborted), { once: true })
      if (ctx.abort.aborted) succeedVoid(aborted)
      succeedVoid(ready)
      return Effect.callback<never>(() => Effect.sync(() => succeedVoid(aborted)))
    }
    const restore = Effect.addFinalizer(() => Effect.sync(() => void (tool.execute = original)))
    return { ready, aborted, restore }
  })
}

noLLMServer.instance(
  "interrupt propagates abort signal to read tool via file part (text/plain)",
  () =>
    Effect.gen(function* () {
      const { directory: dir } = yield* TestInstance
      const registry = yield* ToolRegistry.Service
      const { read } = yield* registry.named()
      const { ready, restore } = yield* hangUntilAborted(read)
      yield* restore

      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Abort Test" })

      const testFile = path.join(dir, "test.txt")
      yield* writeText(testFile, "hello world")

      const fiber = yield* prompt
        .prompt({
          sessionID: chat.id,
          agent: "build",
          parts: [
            { type: "text", text: "read this" },
            { type: "file", url: `file://${testFile}`, filename: "test.txt", mime: "text/plain" },
          ],
        })
        .pipe(Effect.forkChild)

      yield* awaitWithTimeout(Deferred.await(ready), "timed out waiting for read tool to start", "10 seconds")
      yield* prompt.cancel(chat.id)
      yield* Fiber.interrupt(fiber)
      const exit = yield* Fiber.await(fiber)
      expect(Exit.isFailure(exit)).toBe(true)
    }),
  { config: cfg },
  30_000,
)

noLLMServer.instance(
  "interrupt propagates abort signal to read tool via file part (directory)",
  () =>
    Effect.gen(function* () {
      const { directory: dir } = yield* TestInstance
      const registry = yield* ToolRegistry.Service
      const { read } = yield* registry.named()
      const { ready, restore } = yield* hangUntilAborted(read)
      yield* restore

      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({ title: "Abort Test" })

      const fiber = yield* prompt
        .prompt({
          sessionID: chat.id,
          agent: "build",
          parts: [
            { type: "text", text: "read this" },
            { type: "file", url: `file://${dir}`, filename: "dir", mime: "application/x-directory" },
          ],
        })
        .pipe(Effect.forkChild)

      yield* awaitWithTimeout(Deferred.await(ready), "timed out waiting for read tool to start", "10 seconds")
      yield* prompt.cancel(chat.id)
      yield* Fiber.interrupt(fiber)
      const exit = yield* Fiber.await(fiber)
      expect(Exit.isFailure(exit)).toBe(true)
    }),
  { config: cfg },
  30_000,
)

// Missing file handling

noLLMServer.instance(
  "does not fail the prompt when a file part is missing",
  () =>
    Effect.gen(function* () {
      const { directory: dir } = yield* TestInstance
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const session = yield* sessions.create({})

      const missing = path.join(dir, "does-not-exist.ts")
      const msg = yield* prompt.prompt({
        sessionID: session.id,
        agent: "build",
        noReply: true,
        parts: [
          { type: "text", text: "please review @does-not-exist.ts" },
          {
            type: "file",
            mime: "text/plain",
            url: `file://${missing}`,
            filename: "does-not-exist.ts",
          },
        ],
      })

      if (msg.info.role !== "user") throw new Error("expected user message")
      const hasFailure = msg.parts.some(
        (part) => part.type === "text" && part.synthetic && part.text.includes("Read tool failed to read"),
      )
      expect(hasFailure).toBe(true)

      yield* sessions.remove(session.id)
    }),
  { config: cfg },
)

noLLMServer.instance(
  "keeps stored part order stable when file resolution is async",
  () =>
    Effect.gen(function* () {
      const { directory: dir } = yield* TestInstance
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const session = yield* sessions.create({})

      const missing = path.join(dir, "still-missing.ts")
      const msg = yield* prompt.prompt({
        sessionID: session.id,
        agent: "build",
        noReply: true,
        parts: [
          {
            type: "file",
            mime: "text/plain",
            url: `file://${missing}`,
            filename: "still-missing.ts",
          },
          { type: "text", text: "after-file" },
        ],
      })

      if (msg.info.role !== "user") throw new Error("expected user message")

      const stored = yield* MessageV2.get({
        sessionID: session.id,
        messageID: msg.info.id,
      })
      const text = stored.parts.filter((part) => part.type === "text").map((part) => part.text)

      expect(text[0]?.startsWith("Called the Read tool with the following input:")).toBe(true)
      expect(text[1]?.includes("Read tool failed to read")).toBe(true)
      expect(text[2]).toBe("after-file")

      yield* sessions.remove(session.id)
    }),
  { config: cfg },
)

// Special characters in filenames

noLLMServer.instance(
  "handles filenames with # character",
  () =>
    Effect.gen(function* () {
      const { directory: dir } = yield* TestInstance
      yield* writeText(path.join(dir, "file#name.txt"), "special content\n")

      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const session = yield* sessions.create({})
      const parts = yield* prompt.resolvePromptParts("Read @file#name.txt")
      const fileParts = parts.filter((part) => part.type === "file")

      expect(fileParts.length).toBe(1)
      expect(fileParts[0].filename).toBe("file#name.txt")
      expect(fileParts[0].url).toContain("%23")

      const decodedPath = fileURLToPath(fileParts[0].url)
      expect(decodedPath).toBe(path.join(dir, "file#name.txt"))

      const message = yield* prompt.prompt({
        sessionID: session.id,
        parts,
        noReply: true,
      })
      const stored = yield* MessageV2.get({ sessionID: session.id, messageID: message.info.id })
      const textParts = stored.parts.filter((part) => part.type === "text")
      const hasContent = textParts.some((part) => part.text.includes("special content"))
      expect(hasContent).toBe(true)

      yield* sessions.remove(session.id)
    }),
  { git: true, config: cfg },
)

// Regression: empty assistant turn loop

it.instance("does not loop empty assistant turns for a simple reply", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig(providerCfg)
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const session = yield* sessions.create({ title: "Prompt regression" })

    yield* llm.text("packages/opencode/src/session/processor.ts")

    const result = yield* prompt.prompt({
      sessionID: session.id,
      agent: "build",
      parts: [{ type: "text", text: "Where is SessionProcessor?" }],
    })

    expect(result.info.role).toBe("assistant")
    expect(result.parts.some((part) => part.type === "text" && part.text.includes("processor.ts"))).toBe(true)

    const msgs = yield* sessions.messages({ sessionID: session.id })
    expect(msgs.filter((msg) => msg.info.role === "assistant")).toHaveLength(1)
    expect(yield* llm.calls).toBe(1)
  }),
)

it.instance("records aborted errors when prompt is cancelled mid-stream", () =>
  Effect.gen(function* () {
    const { llm } = yield* useServerConfig(providerCfg)
    const prompt = yield* SessionPrompt.Service
    const sessions = yield* Session.Service
    const session = yield* sessions.create({ title: "Prompt cancel regression" })

    yield* llm.hang

    const fiber = yield* prompt
      .prompt({
        sessionID: session.id,
        agent: "build",
        parts: [{ type: "text", text: "Cancel me" }],
      })
      .pipe(Effect.forkChild)

    yield* llm.wait(1)
    yield* waitForBusy(session.id)
    yield* prompt.cancel(session.id)

    const exit = yield* Fiber.await(fiber)
    expect(Exit.isSuccess(exit)).toBe(true)
    if (Exit.isSuccess(exit)) {
      expect(exit.value.info.role).toBe("assistant")
      if (exit.value.info.role === "assistant") {
        expect(exit.value.info.error?.name).toBe("MessageAbortedError")
      }
    }

    const msgs = yield* sessions.messages({ sessionID: session.id })
    const last = msgs.findLast((msg) => msg.info.role === "assistant")
    expect(last?.info.role).toBe("assistant")
    if (last?.info.role === "assistant") {
      expect(last.info.error?.name).toBe("MessageAbortedError")
    }
  }),
)

// Agent variant

noLLMServer.instance(
  "applies agent variant only when using agent model",
  () =>
    Effect.gen(function* () {
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const session = yield* sessions.create({})

      const other = yield* prompt.prompt({
        sessionID: session.id,
        agent: "build",
        model: { providerID: ProviderV2.ID.make("opencode"), modelID: ModelV2.ID.make("kimi-k2.5-free") },
        noReply: true,
        parts: [{ type: "text", text: "hello" }],
      })
      if (other.info.role !== "user") throw new Error("expected user message")
      expect(other.info.model.variant).toBeUndefined()

      const match = yield* prompt.prompt({
        sessionID: session.id,
        agent: "build",
        noReply: true,
        parts: [{ type: "text", text: "hello again" }],
      })
      if (match.info.role !== "user") throw new Error("expected user message")
      expect(match.info.model).toEqual({
        providerID: ProviderV2.ID.make("test"),
        modelID: ModelV2.ID.make("test-model"),
        variant: "xhigh",
      })
      expect(match.info.model.variant).toBe("xhigh")

      const override = yield* prompt.prompt({
        sessionID: session.id,
        agent: "build",
        noReply: true,
        variant: "high",
        parts: [{ type: "text", text: "hello third" }],
      })
      if (override.info.role !== "user") throw new Error("expected user message")
      expect(override.info.model.variant).toBe("high")

      yield* sessions.remove(session.id)
    }),
  {
    config: {
      ...cfg,
      provider: {
        ...cfg.provider,
        test: {
          ...cfg.provider.test,
          models: {
            "test-model": {
              ...cfg.provider.test.models["test-model"],
              variants: { xhigh: {}, high: {} },
            },
          },
        },
      },
      agent: {
        build: {
          model: "test/test-model",
          variant: "xhigh",
        },
      },
    },
  },
)

// Agent / command resolution errors

noLLMServer.instance(
  "unknown agent throws typed error",
  () =>
    Effect.gen(function* () {
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const session = yield* sessions.create({})
      const exit = yield* prompt
        .prompt({
          sessionID: session.id,
          agent: "nonexistent-agent-xyz",
          noReply: true,
          parts: [{ type: "text", text: "hello" }],
        })
        .pipe(Effect.exit)

      expect(Exit.isFailure(exit)).toBe(true)
      if (Exit.isFailure(exit)) {
        const err = Cause.squash(exit.cause)
        expect(err).not.toBeInstanceOf(TypeError)
        expect(NamedError.Unknown.isInstance(err)).toBe(true)
        if (NamedError.Unknown.isInstance(err)) {
          expect(err.data.message).toContain('Agent not found: "nonexistent-agent-xyz"')
        }
      }
    }),
  30_000,
)

noLLMServer.instance(
  "unknown agent error includes available agent names",
  () =>
    Effect.gen(function* () {
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const session = yield* sessions.create({})
      const exit = yield* prompt
        .prompt({
          sessionID: session.id,
          agent: "nonexistent-agent-xyz",
          noReply: true,
          parts: [{ type: "text", text: "hello" }],
        })
        .pipe(Effect.exit)

      expect(Exit.isFailure(exit)).toBe(true)
      if (Exit.isFailure(exit)) {
        const err = Cause.squash(exit.cause)
        expect(NamedError.Unknown.isInstance(err)).toBe(true)
        if (NamedError.Unknown.isInstance(err)) {
          expect(err.data.message).toContain("build")
        }
      }
    }),
  30_000,
)

noLLMServer.instance(
  "unknown command throws typed error with available names",
  () =>
    Effect.gen(function* () {
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const session = yield* sessions.create({})
      const exit = yield* prompt
        .command({
          sessionID: session.id,
          command: "nonexistent-command-xyz",
          arguments: "",
        })
        .pipe(Effect.exit)

      expect(Exit.isFailure(exit)).toBe(true)
      if (Exit.isFailure(exit)) {
        const err = Cause.squash(exit.cause)
        expect(err).not.toBeInstanceOf(TypeError)
        expect(NamedError.Unknown.isInstance(err)).toBe(true)
        if (NamedError.Unknown.isInstance(err)) {
          expect(err.data.message).toContain('Command not found: "nonexistent-command-xyz"')
          expect(err.data.message).toContain("init")
        }
      }
    }),
  30_000,
)
