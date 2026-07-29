import { SessionV1 } from "@opencode-ai/core/v1/session"
import { Database } from "@opencode-ai/core/database/database"
import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { EventV2Bridge } from "@/event-v2-bridge"
import { describe, expect, test } from "bun:test"
import { tool } from "ai"
import { Cause, Effect, Exit, Fiber, Layer, Schema, Stream } from "effect"
import path from "path"
import z from "zod"
import { Agent } from "../../src/agent/agent"
import { Provider } from "@/provider/provider"

import { Session } from "@/session/session"
import { LLM } from "../../src/session/llm"
import { MessageV2 } from "../../src/session/message-v2"
import { SessionProcessor } from "../../src/session/processor"
import { classifyProviderFailure } from "../../src/session/processor"
import { validateFallbackCandidates } from "../../src/session/routing"
import { MessageID, PartID, SessionID } from "../../src/session/schema"
import { SessionStatus } from "../../src/session/status"
import { SessionSummary } from "../../src/session/summary"
import { CrossSpawnSpawner } from "@opencode-ai/core/cross-spawn-spawner"
import { provideTmpdirInstance, provideTmpdirServer } from "../fixture/fixture"
import { awaitWithTimeout, testEffect } from "../lib/effect"
import { raw, reply, TestLLMServer } from "../lib/llm-server"
import { RuntimeFlags } from "@/effect/runtime-flags"
import { ProviderV2 } from "@opencode-ai/core/provider"
import { ModelV2 } from "@opencode-ai/core/model"
import { SessionProjector } from "@opencode-ai/core/session/projector"
import { LLMEvent } from "@opencode-ai/llm"
import { Plugin } from "@/plugin"
import { PermissionV1 } from "@opencode-ai/core/v1/permission"
import { Question } from "@/question"
import * as SessionTools from "@/session/tools"
import { ToolRegistry } from "@/tool/registry"
import { Permission } from "@/permission"
import { MCP } from "@/mcp"
import { Truncate } from "@/tool/truncate"
import { Config } from "@/config/config"

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
    fallback: {
      name: "Fallback",
      id: "fallback",
      env: [],
      npm: "@ai-sdk/openai-compatible",
      models: {
        "fallback-model": {
          id: "fallback-model",
          name: "Fallback Model",
          attachment: false,
          reasoning: false,
          temperature: false,
          tool_call: true,
          release_date: "2025-01-01",
          limit: { context: 100000, output: 10000 },
          cost: { input: 0, output: 0 },
          options: {},
          variants: { secondary: {} },
        },
      },
      options: {
        apiKey: "test-key",
        baseURL: "http://localhost:1/v1",
      },
    },
  },
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

function agent(): Agent.Info {
  return {
    name: "build",
    mode: "primary",
    options: {},
    permission: [{ permission: "*", pattern: "*", action: "allow" }],
  }
}

function defer<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void
  const promise = new Promise<T>((done) => {
    resolve = done
  })
  return { promise, resolve }
}

const waitFor = <A, E, R>(check: Effect.Effect<A | undefined, E, R>, message: string) =>
  Effect.gen(function* () {
    const stop = Date.now() + 500
    while (Date.now() < stop) {
      const value = yield* check
      if (value !== undefined) return value
      yield* Effect.sleep("10 millis")
    }
    return yield* Effect.fail(new Error(message))
  })

const user = Effect.fn("TestSession.user")(function* (sessionID: SessionID, text: string) {
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

const assistant = Effect.fn("TestSession.assistant")(function* (
  sessionID: SessionID,
  parentID: MessageID,
  root: string,
) {
  const session = yield* Session.Service
  const msg: SessionV1.Assistant = {
    id: MessageID.ascending(),
    role: "assistant",
    sessionID,
    mode: "build",
    agent: "build",
    path: { cwd: root, root },
    cost: 0,
    tokens: {
      total: 0,
      input: 0,
      output: 0,
      reasoning: 0,
      cache: { read: 0, write: 0 },
    },
    modelID: ref.modelID,
    providerID: ref.providerID,
    parentID,
    time: { created: Date.now() },
    finish: "end_turn",
  }
  yield* session.updateMessage(msg)
  return msg
})

const root = LayerNode.group([
  SessionProcessor.node,
  Session.node,
  SessionProjector.node,
  Provider.node,
  Database.node,
  EventV2Bridge.node,
  SessionStatus.node,
  CrossSpawnSpawner.node,
])
const replacements = [
  [SessionSummary.node, summary],
  [RuntimeFlags.node, RuntimeFlags.layer({ experimentalEventSystem: true })],
] as const
const env = LayerNode.compile(
  LayerNode.group([root, LayerNode.make({ service: TestLLMServer, layer: TestLLMServer.layer, deps: [] })]),
  replacements,
)

const it = testEffect(env)

let routedFallbackModels: Array<{ providerID: string; modelID: string; variant?: string }> = [
  { providerID: "fallback", modelID: "fallback-model", variant: "secondary" },
]
const routedFallbackPlugin = Layer.mock(Plugin.Service, {
  list: () => Effect.succeed([]),
  trigger: ((_name: unknown, _input: unknown, output: unknown) =>
    Effect.succeed(output)) as Plugin.Interface["trigger"],
  triggerProviderFailure: () => Effect.succeed({ action: "fallback" as const, models: routedFallbackModels }),
})
const routedStopPlugin = Layer.mock(Plugin.Service, {
  list: () => Effect.succeed([]),
  trigger: ((_name: unknown, _input: unknown, output: unknown) =>
    Effect.succeed(output)) as Plugin.Interface["trigger"],
  triggerProviderFailure: () => Effect.succeed({ action: "stop" as const, models: [] }),
})
const unhandledPlugin = Layer.mock(Plugin.Service, {
  list: () => Effect.succeed([]),
  trigger: ((_name: unknown, _input: unknown, output: unknown) =>
    Effect.succeed(output)) as Plugin.Interface["trigger"],
  triggerProviderFailure: () => Effect.succeed({ action: "unhandled" as const, models: [] }),
})
const throwingFailurePlugin = Layer.mock(Plugin.Service, {
  list: () => Effect.succeed([]),
  trigger: ((_name: unknown, _input: unknown, output: unknown) =>
    Effect.succeed(output)) as Plugin.Interface["trigger"],
  triggerProviderFailure: () => Effect.fail("provider failure hook exploded"),
})
const routedFallbackEnv = LayerNode.compile(
  LayerNode.group([root, LayerNode.make({ service: TestLLMServer, layer: TestLLMServer.layer, deps: [] })]),
  [...replacements, [Plugin.node, routedFallbackPlugin]],
)
const routedStopEnv = LayerNode.compile(
  LayerNode.group([root, LayerNode.make({ service: TestLLMServer, layer: TestLLMServer.layer, deps: [] })]),
  [...replacements, [Plugin.node, routedStopPlugin]],
)
const unhandledEnv = LayerNode.compile(
  LayerNode.group([root, LayerNode.make({ service: TestLLMServer, layer: TestLLMServer.layer, deps: [] })]),
  [...replacements, [Plugin.node, unhandledPlugin]],
)
const throwingFailureEnv = LayerNode.compile(
  LayerNode.group([root, LayerNode.make({ service: TestLLMServer, layer: TestLLMServer.layer, deps: [] })]),
  [...replacements, [Plugin.node, throwingFailurePlugin]],
)
const itRoutedFallback = testEffect(routedFallbackEnv)
const itRoutedStop = testEffect(routedStopEnv)
const itUnhandled = testEffect(unhandledEnv)
const itThrowingFailure = testEffect(throwingFailureEnv)

const providerErrorLLM = Layer.succeed(
  LLM.Service,
  LLM.Service.of({
    stream: () =>
      Stream.make(
        LLMEvent.stepStart({ index: 0 }),
        LLMEvent.toolInputStart({ id: "call-1", name: "lookup" }),
        LLMEvent.toolInputEnd({ id: "call-1", name: "lookup" }),
        LLMEvent.toolCall({ id: "call-1", name: "lookup", input: {}, providerExecuted: true }),
        LLMEvent.toolResult({
          id: "call-1",
          name: "lookup",
          result: { type: "error", value: "provider boom" },
          providerExecuted: true,
        }),
        LLMEvent.stepFinish({ index: 0, reason: "stop" }),
        LLMEvent.finish({ reason: "stop" }),
      ),
  }),
)
const providerErrorEnv = LayerNode.compile(root, [...replacements, [LLM.node, providerErrorLLM]])
const itProviderError = testEffect(providerErrorEnv)

const fragmentFailureLLM = Layer.succeed(
  LLM.Service,
  LLM.Service.of({
    stream: () =>
      Stream.make(
        LLMEvent.stepStart({ index: 0 }),
        LLMEvent.reasoningStart({ id: "reasoning-1" }),
        LLMEvent.reasoningDelta({ id: "reasoning-1", text: "thinking" }),
        LLMEvent.textStart({ id: "text-1" }),
        LLMEvent.textDelta({ id: "text-1", text: "partial" }),
        LLMEvent.providerError({ message: "provider boom" }),
      ),
  }),
)
const fragmentFailureEnv = LayerNode.compile(root, [...replacements, [LLM.node, fragmentFailureLLM]])
const itFragmentFailure = testEffect(fragmentFailureEnv)

let lifecycleStream: () => Stream.Stream<LLMEvent, unknown, never> = () => Stream.never
const lifecycleLLM = Layer.succeed(
  LLM.Service,
  LLM.Service.of({
    stream: () => lifecycleStream(),
  }),
)
const lifecycleRoot = LayerNode.group([
  root,
  LayerNode.make({ service: TestLLMServer, layer: TestLLMServer.layer, deps: [] }),
  ToolRegistry.node,
  Permission.node,
  MCP.node,
  Truncate.node,
  Config.node,
  Agent.node,
])
const lifecycleEnv = LayerNode.compile(lifecycleRoot, [...replacements, [LLM.node, lifecycleLLM]])
const lifecycleRoutedEnv = LayerNode.compile(
  LayerNode.group([root, LayerNode.make({ service: TestLLMServer, layer: TestLLMServer.layer, deps: [] })]),
  [...replacements, [LLM.node, lifecycleLLM], [Plugin.node, routedFallbackPlugin]],
)
const itLifecycle = testEffect(lifecycleEnv)
const itLifecycleRouted = testEffect(lifecycleRoutedEnv)

let wrapperExecute: () => Effect.Effect<
  { title: string; metadata: Record<string, unknown>; output: string },
  Error
> = () => Effect.succeed({ title: "wrapper", metadata: {}, output: "wrapper result" })
const wrapperRegistry = Layer.mock(ToolRegistry.Service, {
  tools: () =>
    Effect.succeed([
      {
        id: "lifecycle",
        description: "Lifecycle test tool",
        parameters: Schema.Struct({}),
        execute: () => wrapperExecute(),
      },
    ] as never),
})
const wrapperServices = Layer.mergeAll(
  wrapperRegistry,
  routedFallbackPlugin,
  Layer.mock(Agent.Service, { get: () => Effect.succeed(agent()), list: () => Effect.succeed([agent()]) }),
  Layer.mock(Permission.Service, { ask: () => Effect.void }),
  Layer.mock(MCP.Service, { clients: () => Effect.succeed({}), tools: () => Effect.succeed({}) }),
  Layer.mock(Truncate.Service, { output: (text: string) => Effect.succeed({ content: text, truncated: false }) }),
  RuntimeFlags.layer({ experimentalEventSystem: true }),
)
const itWrapper = itLifecycle

const toolStart = (id = "call-1", providerExecuted = false) =>
  Stream.make(
    LLMEvent.toolInputStart({ id, name: "lookup" }),
    LLMEvent.toolInputEnd({ id, name: "lookup" }),
    LLMEvent.toolCall({ id, name: "lookup", input: { query: "test" }, providerExecuted }),
  )

function waitThen<T, E>(wait: Promise<void>, stream: Stream.Stream<T, E, never>) {
  return Stream.fromEffect(Effect.promise(() => wait)).pipe(Stream.flatMap(() => stream))
}

const provideLifecycle = <A, E, R>(self: (dir: string) => Effect.Effect<A, E, R>) =>
  provideTmpdirInstance(self, { config: cfg })

const promptOps = {
  cancel: () => Effect.void,
  resolvePromptParts: () => Effect.succeed([]),
  prompt: () => Effect.die("unused prompt operation"),
}

function streamInput(parent: SessionV1.User, chatID: SessionID, model: Provider.Model, text: string) {
  return {
    user: parent,
    sessionID: chatID,
    model,
    agent: agent(),
    system: [],
    messages: [{ role: "user" as const, content: text }],
    tools: {},
  }
}

const boot = Effect.fn("test.boot")(function* () {
  const processors = yield* SessionProcessor.Service
  const session = yield* Session.Service
  const provider = yield* Provider.Service
  return { processors, session, provider }
})

const beginLifecycleTool = Effect.fn("TestSession.beginLifecycleTool")(function* (
  dir: string,
  text: string,
  joinToolsTimeout?: "50 millis",
) {
  const { processors, session, provider } = yield* boot()
  const chat = yield* session.create({})
  const parent = yield* user(chat.id, text)
  const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
  const model = yield* provider.getModel(ref.providerID, ref.modelID)
  const handle = yield* processors.create({
    assistantMessage: msg,
    sessionID: chat.id,
    model,
    provider,
    ...(joinToolsTimeout ? { joinToolsTimeout } : {}),
  })
  return { chat, parent, msg, model, handle }
})

const waitForTool = (messageID: MessageID, callID = "call-1") =>
  waitFor(
    MessageV2.parts(messageID).pipe(
      Effect.map((parts) =>
        parts.find((part): part is SessionV1.ToolPart => part.type === "tool" && part.callID === callID),
      ),
    ),
    `timed out waiting for tool ${callID}`,
  )

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("session.processor provider failure classifier", () => {
  test("classifies serialized provider errors after lossy normalization", () => {
    const api = (statusCode: number, isRetryable = false) =>
      new SessionV1.APIError({ message: "provider failure", statusCode, isRetryable }).toObject()
    const normalize = (error: unknown) =>
      MessageV2.fromError(error, { providerID: ProviderV2.ID.make("test-provider") })

    expect(classifyProviderFailure(new SessionV1.AuthError({ providerID: "test", message: "denied" }).toObject())).toBe(
      "auth",
    )
    expect(classifyProviderFailure(normalize(api(429)))).toBeUndefined()
    expect(classifyProviderFailure(api(429))).toBe("rate_limit")
    expect(classifyProviderFailure(api(500))).toBe("server")
    expect(classifyProviderFailure(api(0, true))).toBe("network")
    expect(classifyProviderFailure(new SessionV1.ContextOverflowError({ message: "too large" }).toObject())).toBe(
      "overflow",
    )
    expect(classifyProviderFailure(new SessionV1.AbortedError({ message: "cancelled" }).toObject())).toBeUndefined()
    expect(classifyProviderFailure(normalize(new Error("invariant")))).toBeUndefined()
  })
})

describe("session routing candidate identity", () => {
  test("keeps slash-containing tuples distinct", () => {
    const candidates = [
      { providerID: "fallback/one", modelID: "model" },
      { providerID: "fallback", modelID: "one/model" },
    ]

    expect(validateFallbackCandidates({ providerID: "current", modelID: "model" }, candidates)).toEqual(candidates)
  })
})

for (const [failure, error] of [
  ["401", new SessionV1.APIError({ message: "provider body", statusCode: 401, isRetryable: false }).toObject()],
  ["429", new SessionV1.APIError({ message: "provider body", statusCode: 429, isRetryable: false }).toObject()],
  ["500", new SessionV1.APIError({ message: "provider body", statusCode: 500, isRetryable: false }).toObject()],
  ["network", new SessionV1.APIError({ message: "provider body", isRetryable: true }).toObject()],
] as const) {
  for (const [proposal, models, valid] of [
    [
      "unresolved provider/model",
      [
        { providerID: "fallback", modelID: "fallback-model" },
        { providerID: "missing", modelID: "missing-model" },
      ],
      false,
    ],
    [
      "invalid variant",
      [
        { providerID: "fallback", modelID: "fallback-model" },
        { providerID: "fallback", modelID: "fallback-model", variant: "missing" },
      ],
      false,
    ],
    [
      "duplicate candidate",
      [
        { providerID: "fallback", modelID: "fallback-model", variant: "secondary" },
        { providerID: "fallback", modelID: "fallback-model", variant: "secondary" },
      ],
      false,
    ],
    [
      "current candidate",
      [
        { providerID: "fallback", modelID: "fallback-model" },
        { providerID: "test", modelID: "test-model" },
      ],
      false,
    ],
    ["valid explicit variant", [{ providerID: "fallback", modelID: "fallback-model", variant: "secondary" }], true],
    ["valid absent variant", [{ providerID: "fallback", modelID: "fallback-model" }], true],
  ] as const) {
    itLifecycleRouted.live(`${failure} ${proposal} ${valid ? "creates" : "rejects"} a provider handoff`, () =>
      provideLifecycle((dir) =>
        Effect.gen(function* () {
          const previous = routedFallbackModels
          routedFallbackModels = [...models]
          yield* Effect.addFinalizer(() =>
            Effect.sync(() => {
              routedFallbackModels = previous
            }),
          )
          lifecycleStream = () => Stream.fail(error)
          const { chat, parent, model, handle } = yield* beginLifecycleTool(dir, `${failure} ${proposal}`)
          const result = yield* handle.process(streamInput(parent, chat.id, model, `${failure} ${proposal}`))

          if (valid) {
            expect(result).toMatchObject({ type: "handoff" })
            expect(JSON.stringify(handle.message.routedHandoff?.next)).toContain(JSON.stringify(models[0]))
            return
          }

          expect(result).toBe("stop")
          expect(handle.message.routedHandoff).toBeUndefined()
          expect(JSON.stringify(handle.message.error)).toContain("Routed provider failure: invalid fallback proposal")
          expect(JSON.stringify(handle.message.error)).not.toContain("provider body")
        }),
      ),
    )
  }
}

itLifecycle.live("finalizer completion wins over a late stream result exactly once", () =>
  provideLifecycle((dir) =>
    Effect.gen(function* () {
      const gate = defer<void>()
      lifecycleStream = () =>
        Stream.concat(
          toolStart(),
          waitThen(
            gate.promise,
            Stream.make(
              LLMEvent.toolResult({
                id: "call-1",
                name: "lookup",
                result: { type: "text", value: "late stream result" },
              }),
              LLMEvent.finish({ reason: "stop" }),
            ),
          ),
        )
      const { chat, parent, msg, model, handle } = yield* beginLifecycleTool(dir, "finalizer")
      const run = yield* handle.process(streamInput(parent, chat.id, model, "finalizer")).pipe(Effect.forkChild)
      yield* waitForTool(msg.id)
      yield* handle.startToolCall("call-1")
      yield* handle.completeToolCall("call-1", { title: "final", metadata: {}, output: "finalizer result" })
      gate.resolve()
      yield* awaitWithTimeout(Fiber.join(run), "timed out waiting for late result", "500 millis")

      const call = yield* waitForTool(msg.id)
      expect(call.state.status).toBe("completed")
      if (call.state.status === "completed") expect(call.state.output).toBe("finalizer result")
    }),
  ),
)

itLifecycle.live("finalizer failures release throws and preserve readable errors", () =>
  provideLifecycle((dir) =>
    Effect.gen(function* () {
      lifecycleStream = () => Stream.concat(toolStart(), Stream.never)
      const { chat, parent, msg, model, handle } = yield* beginLifecycleTool(dir, "throw")
      const run = yield* handle.process(streamInput(parent, chat.id, model, "throw")).pipe(Effect.forkChild)
      yield* waitForTool(msg.id)
      yield* handle.startToolCall("call-1")
      yield* handle.failToolCall("call-1", new Error("execute exploded"))
      yield* Fiber.interrupt(run)

      const call = yield* waitForTool(msg.id)
      expect(call.state.status).toBe("error")
      if (call.state.status === "error") expect(call.state.error).toBe("execute exploded")
    }),
  ),
)

itWrapper.live(
  "interrupted local execution finalizer persists interrupted metadata before cleanup",
  () =>
    provideLifecycle((dir) =>
      Effect.gen(function* () {
        lifecycleStream = () => Stream.concat(toolStart(), Stream.never)
        wrapperExecute = () => Effect.sleep("10 millis").pipe(Effect.andThen(Effect.interrupt))
        const { chat, parent, msg, model, handle } = yield* beginLifecycleTool(dir, "interrupted local execute")
        const run = yield* handle
          .process(streamInput(parent, chat.id, model, "interrupted local execute"))
          .pipe(Effect.forkChild)
        yield* waitForTool(msg.id)
        const tools = yield* SessionTools.resolve({
          agent: agent(),
          model,
          session: chat,
          processor: handle,
          bypassAgentCheck: false,
          messages: [],
          promptOps,
        }).pipe(Effect.provide(wrapperServices))
        const result = yield* Effect.promise(() =>
          tools.lifecycle.execute!(
            {},
            { toolCallId: "call-1", abortSignal: new AbortController().signal, messages: [] },
          ),
        ).pipe(Effect.exit)
        expect(Exit.isFailure(result)).toBe(true)
        const call = yield* waitFor(
          MessageV2.parts(msg.id).pipe(
            Effect.map((parts) =>
              parts.find(
                (part): part is SessionV1.ToolPart =>
                  part.type === "tool" && part.callID === "call-1" && part.state.status === "error",
              ),
            ),
          ),
          "timed out waiting for interrupted local execution finalizer",
        )
        expect(call.state.status).toBe("error")
        if (call.state.status === "error") {
          expect(call.state.metadata?.interrupted).toBe(true)
        }
        yield* Fiber.interrupt(run)
      }),
    ) as never,
)

for (const [name, execute, expected] of [
  ["success", () => Effect.succeed({ title: "wrapper", metadata: {}, output: "wrapper result" }), "completed"],
  ["throw", () => Effect.fail(new Error("wrapper exploded")), "error"],
] as const) {
  itWrapper.live(
    `SessionTools.resolve ${name} settles without a stream tool result`,
    () =>
      provideLifecycle((dir) =>
        Effect.gen(function* () {
          lifecycleStream = () => Stream.concat(toolStart(), Stream.never)
          wrapperExecute = execute
          const { chat, parent, msg, model, handle } = yield* beginLifecycleTool(dir, `wrapper ${name}`)
          const run = yield* handle
            .process(streamInput(parent, chat.id, model, `wrapper ${name}`))
            .pipe(Effect.forkChild)
          yield* waitForTool(msg.id)
          const tools = yield* SessionTools.resolve({
            agent: agent(),
            model,
            session: chat,
            processor: handle,
            bypassAgentCheck: false,
            messages: [],
            promptOps,
          }).pipe(Effect.provide(wrapperServices))
          const result = yield* Effect.promise(() =>
            tools.lifecycle.execute!(
              {},
              { toolCallId: "call-1", abortSignal: new AbortController().signal, messages: [] },
            ),
          ).pipe(Effect.exit)
          expect(Exit.isSuccess(result)).toBe(name === "success")
          yield* Fiber.interrupt(run)

          const call = yield* waitForTool(msg.id)
          expect(call.state.status).toBe(expected)
          if (call.state.status === "completed") expect(call.state.output).toBe("wrapper result")
          if (call.state.status === "error") expect(call.state.error).toBe("wrapper exploded")
        }),
      ) as never,
  )
}

itLifecycle.live("failed tool updates remain registered for interrupted cleanup", () =>
  provideLifecycle((dir) =>
    Effect.gen(function* () {
      lifecycleStream = () => Stream.concat(toolStart(), Stream.never)
      const { chat, parent, msg, model, handle } = yield* beginLifecycleTool(dir, "update")
      const run = yield* handle.process(streamInput(parent, chat.id, model, "update")).pipe(Effect.forkChild)
      yield* waitForTool(msg.id)
      yield* handle.startToolCall("call-1")
      expect(
        Exit.isFailure(
          yield* handle
            .updateToolCall("call-1", () => {
              throw new Error("update failed")
            })
            .pipe(Effect.exit),
        ),
      ).toBe(true)
      yield* Fiber.interrupt(run)
      const call = yield* waitForTool(msg.id)
      expect(call.state.status).toBe("error")
      if (call.state.status === "error") expect(call.state.metadata?.interrupted).toBe(true)
    }),
  ),
)

for (const [name, error] of [
  ["permission", new PermissionV1.RejectedError()],
  ["question", new Question.RejectedError()],
] as const) {
  itLifecycle.live(`${name} rejection blocks the processor after Cause normalization`, () =>
    provideLifecycle((dir) =>
      Effect.gen(function* () {
        const gate = defer<void>()
        lifecycleStream = () =>
          Stream.concat(toolStart(), waitThen(gate.promise, Stream.make(LLMEvent.finish({ reason: "stop" }))))
        const { chat, parent, msg, model, handle } = yield* beginLifecycleTool(dir, name)
        const run = yield* handle.process(streamInput(parent, chat.id, model, name)).pipe(Effect.forkChild)
        yield* waitForTool(msg.id)
        yield* handle.startToolCall("call-1")
        yield* handle.failToolCall("call-1", error)
        gate.resolve()
        expect(yield* awaitWithTimeout(Fiber.join(run), `timed out waiting for ${name} rejection`, "500 millis")).toBe(
          "stop",
        )

        const call = yield* waitForTool(msg.id)
        expect(call.state.status).toBe("error")
        if (call.state.status === "error") {
          expect(call.state.error).toBe(error.message)
          expect(call.state.metadata?.interrupted).toBeUndefined()
        }
      }),
    ),
  )
}

itLifecycle.live("finalizer image normalization omits failed images", () =>
  provideLifecycle((dir) =>
    Effect.gen(function* () {
      lifecycleStream = () => Stream.concat(toolStart(), Stream.never)
      const { chat, parent, msg, model, handle } = yield* beginLifecycleTool(dir, "image")
      const run = yield* handle.process(streamInput(parent, chat.id, model, "image")).pipe(Effect.forkChild)
      yield* waitForTool(msg.id)
      yield* handle.startToolCall("call-1")
      yield* handle.completeToolCall("call-1", {
        title: "image",
        metadata: {},
        output: "output",
        attachments: [
          {
            id: PartID.ascending(),
            sessionID: chat.id,
            messageID: msg.id,
            type: "file",
            mime: "image/png",
            url: "data:image/png;base64,not-an-image",
          },
        ],
      })
      yield* Fiber.interrupt(run)

      const call = yield* waitForTool(msg.id)
      expect(call.state.status).toBe("completed")
      if (call.state.status === "completed") {
        expect(call.state.attachments).toBeUndefined()
        expect(call.state.output).toContain("1 image omitted")
      }
    }),
  ),
)

itLifecycleRouted.live("slow client tools complete before a routed handoff", () =>
  provideLifecycle((dir) =>
    Effect.gen(function* () {
      const gate = defer<void>()
      lifecycleStream = () =>
        Stream.concat(
          toolStart(),
          waitThen(
            gate.promise,
            Stream.fail(
              new SessionV1.APIError({ message: "provider failed", statusCode: 429, isRetryable: true }).toObject(),
            ),
          ),
        )
      const { chat, parent, msg, model, handle } = yield* beginLifecycleTool(dir, "slow")
      const run = yield* handle.process(streamInput(parent, chat.id, model, "slow")).pipe(Effect.forkChild)
      yield* waitForTool(msg.id)
      yield* handle.startToolCall("call-1")
      let completions = 0
      yield* Effect.forkScoped(
        Effect.sleep("300 millis").pipe(
          Effect.andThen(() => {
            completions++
            return handle.completeToolCall("call-1", { title: "slow", metadata: {}, output: "slow result" })
          }),
        ),
      )
      const started = Date.now()
      gate.resolve()
      const result = yield* awaitWithTimeout(Fiber.join(run), "timed out waiting for slow client tool", "2 seconds")
      expect(result).toMatchObject({ type: "handoff", failure: "rate_limit" })
      expect(Date.now() - started).toBeGreaterThanOrEqual(250)
      expect(completions).toBe(1)

      const call = yield* waitForTool(msg.id)
      expect(call.state.status).toBe("completed")
      if (call.state.status === "completed") expect(call.state.output).toBe("slow result")
      expect(handle.message.routedHandoff).toMatchObject({ status: "pending", failure: "rate_limit" })
    }),
  ),
)

itLifecycleRouted.live("cancellation during a routed join remains terminal after late completion", () =>
  provideLifecycle((dir) =>
    Effect.gen(function* () {
      const gate = defer<void>()
      lifecycleStream = () =>
        Stream.concat(
          toolStart(),
          waitThen(
            gate.promise,
            Stream.fail(
              new SessionV1.APIError({ message: "provider failed", statusCode: 429, isRetryable: true }).toObject(),
            ),
          ),
        )
      const { chat, parent, msg, model, handle } = yield* beginLifecycleTool(dir, "cancel")
      const run = yield* handle.process(streamInput(parent, chat.id, model, "cancel")).pipe(Effect.forkChild)
      yield* waitForTool(msg.id)
      yield* handle.startToolCall("call-1")
      gate.resolve()
      yield* Effect.sleep("20 millis")
      yield* Fiber.interrupt(run)
      yield* handle.completeToolCall("call-1", { title: "late", metadata: {}, output: "late completion" })

      const call = yield* waitForTool(msg.id)
      expect(call.state.status).toBe("error")
      if (call.state.status === "error") expect(call.state.metadata?.interrupted).toBe(true)
      expect(handle.message.routedHandoff).toBeUndefined()
      expect(handle.message.error?.name).toBe("MessageAbortedError")
    }),
  ),
)

itLifecycleRouted.live("provider-owned running calls fail closed without a join wait", () =>
  provideLifecycle((dir) =>
    Effect.gen(function* () {
      lifecycleStream = () =>
        Stream.concat(
          toolStart("call-1", true),
          Stream.fail(
            new SessionV1.APIError({ message: "provider failed", statusCode: 429, isRetryable: true }).toObject(),
          ),
        )
      const { chat, parent, msg, model, handle } = yield* beginLifecycleTool(dir, "provider")
      const started = Date.now()
      expect(
        yield* awaitWithTimeout(
          handle.process(streamInput(parent, chat.id, model, "provider")),
          "timed out waiting for provider-owned indeterminate stop",
          "500 millis",
        ),
      ).toBe("stop")
      expect(Date.now() - started).toBeLessThan(250)

      const call = yield* waitForTool(msg.id)
      expect(call.state.status).toBe("error")
      expect(handle.message.routedHandoff).toBeUndefined()
      expect(JSON.stringify(handle.message.error)).toContain("Routed provider failure: tool state is indeterminate")
    }),
  ),
)

itLifecycleRouted.live("client tool join timeout fails closed without a handoff", () =>
  provideLifecycle((dir) =>
    Effect.gen(function* () {
      const gate = defer<void>()
      lifecycleStream = () =>
        Stream.concat(
          toolStart(),
          waitThen(
            gate.promise,
            Stream.fail(
              new SessionV1.APIError({ message: "provider failed", statusCode: 429, isRetryable: true }).toObject(),
            ),
          ),
        )
      const { chat, parent, msg, model, handle } = yield* beginLifecycleTool(dir, "timeout", "50 millis")
      const run = yield* handle.process(streamInput(parent, chat.id, model, "timeout")).pipe(Effect.forkChild)
      yield* waitForTool(msg.id)
      yield* handle.startToolCall("call-1")
      gate.resolve()

      expect(yield* awaitWithTimeout(Fiber.join(run), "timed out waiting for client tool stop", "500 millis")).toBe(
        "stop",
      )
      expect(handle.message.routedHandoff).toBeUndefined()
      expect(JSON.stringify(handle.message.error)).toContain("Routed provider failure: tool state is indeterminate")
    }),
  ),
)

const nativeRetry = (statusCode: 429 | 500) =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const { processors, session, provider } = yield* boot()
        const events = yield* EventV2Bridge.Service
        const retries: number[] = []
        yield* llm.error(statusCode, { error: "retry me" })
        yield* llm.error(statusCode, { error: "retry me" })
        yield* llm.hang

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, `native ${statusCode}`)
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const model = yield* provider.getModel(ref.providerID, ref.modelID)
        const off = yield* events.listen((evt) => {
          if (evt.type !== SessionStatus.Event.Status.type) return Effect.void
          const data = evt.data as typeof SessionStatus.Event.Status.data.Type
          if (data.sessionID === chat.id && data.status.type === "retry") retries.push(data.status.attempt)
          return Effect.void
        })
        const handle = yield* processors.create({ assistantMessage: msg, sessionID: chat.id, model, provider })
        const run = yield* handle
          .process(streamInput(parent, chat.id, model, `native ${statusCode}`))
          .pipe(Effect.forkChild)
        yield* awaitWithTimeout(llm.wait(2), `timed out waiting for native ${statusCode} retry`, "5 seconds")
        yield* Fiber.interrupt(run)
        yield* off

        const inputs = yield* llm.inputs
        expect(inputs).toHaveLength(2)
        expect(inputs.every((input) => input.model === "test-model")).toBe(true)
        expect(retries).toEqual([1])
        expect(handle.message.routedHandoff).toBeUndefined()
        const stored = yield* MessageV2.get({ sessionID: chat.id, messageID: msg.id })
        if (stored.info.role === "assistant") expect(stored.info.routedHandoff).toBeUndefined()
      }),
    { config: (url) => providerCfg(url) },
  )

for (const [name, itNative] of [
  ["plugin absent", it],
  ["plugin unhandled", itUnhandled],
] as const) {
  for (const statusCode of [429, 500] as const) {
    itNative.live(`${name} preserves native ${statusCode} retry without a routed handoff`, () =>
      nativeRetry(statusCode),
    )
  }
}

itRoutedFallback.live("routes a 429 before native retry and persists a redacted handoff", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const { processors, session, provider } = yield* boot()
        const status = yield* SessionStatus.Service
        yield* llm.error(429, { error: "provider body must not be in handoff" })

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "route")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const model = yield* provider.getModel(ref.providerID, ref.modelID)
        const handle = yield* processors.create({ assistantMessage: msg, sessionID: chat.id, model, provider })
        const result = yield* handle.process(streamInput(parent, chat.id, model, "route"))
        const stored = yield* MessageV2.get({ sessionID: chat.id, messageID: msg.id })

        expect(result).toMatchObject({ type: "handoff", failure: "rate_limit" })
        expect(yield* llm.calls).toBe(1)
        expect((yield* status.get(chat.id)).type).toBe("busy")
        expect(stored.info.role).toBe("assistant")
        if (stored.info.role !== "assistant") return
        expect(stored.info.routedHandoff).toMatchObject({
          status: "pending",
          failure: "rate_limit",
          from: { providerID: "test", modelID: "test-model" },
          next: { providerID: "fallback", modelID: "fallback-model", variant: "secondary" },
          userMessageID: parent.id,
        })
        expect(JSON.stringify(stored.info.routedHandoff)).not.toContain("provider body")
      }),
    { config: (url) => providerCfg(url) },
  ),
)

itRoutedStop.live("stops routed failures with a stable leak-free diagnostic", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const { processors, session, provider } = yield* boot()
        yield* llm.error(401, { error: "provider body must not leak" })

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "stop")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const model = yield* provider.getModel(ref.providerID, ref.modelID)
        const handle = yield* processors.create({ assistantMessage: msg, sessionID: chat.id, model })

        expect(yield* handle.process(streamInput(parent, chat.id, model, "stop"))).toBe("stop")
        expect(JSON.stringify(handle.message.error)).toContain("Routed provider failure: routing stopped")
        expect(JSON.stringify(handle.message.error)).not.toContain("provider body")
        expect(yield* llm.calls).toBe(1)
      }),
    { config: (url) => providerCfg(url) },
  ),
)

itThrowingFailure.live("provider failure hook exceptions stop without native retry", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const { processors, session, provider } = yield* boot()
        yield* llm.error(429, { error: "must not retry" })
        yield* llm.error(429, { error: "must not retry" })
        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "throwing provider hook")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const model = yield* provider.getModel(ref.providerID, ref.modelID)
        const handle = yield* processors.create({ assistantMessage: msg, sessionID: chat.id, model, provider })

        expect(yield* handle.process(streamInput(parent, chat.id, model, "throwing provider hook"))).toBe("stop")
        expect(yield* llm.calls).toBe(1)
        expect(handle.message.routedHandoff).toBeUndefined()
        expect(JSON.stringify(handle.message.error)).toContain("Routed provider failure: routing stopped")
      }),
    { config: (url) => providerCfg(url) },
  ),
)

it.live("session.processor effect tests capture llm input cleanly", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const database = yield* Database.Service
        const { processors, session, provider } = yield* boot()

        yield* llm.text("hello")

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "hi")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
          provider,
        })

        const input = {
          user: {
            id: parent.id,
            sessionID: chat.id,
            role: "user",
            time: parent.time,
            agent: parent.agent,
            model: { providerID: ref.providerID, modelID: ref.modelID },
          } satisfies SessionV1.User,
          sessionID: chat.id,
          model: mdl,
          agent: agent(),
          system: [],
          messages: [{ role: "user", content: "hi" }],
          tools: {},
        } satisfies LLM.StreamInput

        const value = yield* handle.process(input)
        const parts = yield* MessageV2.parts(msg.id)
        const calls = yield* llm.calls

        expect(value).toBe("continue")
        expect(calls).toBe(1)
        expect(parts.some((part) => part.type === "text" && part.text === "hello")).toBe(true)
      }),
    { config: (url) => providerCfg(url) },
  ),
)

it.live("session.processor effect tests preserve text start time", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const database = yield* Database.Service
        const gate = defer<void>()
        const { processors, session, provider } = yield* boot()

        yield* llm.push(
          raw({
            head: [
              {
                id: "chatcmpl-test",
                object: "chat.completion.chunk",
                choices: [{ delta: { role: "assistant" } }],
              },
              {
                id: "chatcmpl-test",
                object: "chat.completion.chunk",
                choices: [{ delta: { content: "hello" } }],
              },
            ],
            wait: gate.promise,
            tail: [
              {
                id: "chatcmpl-test",
                object: "chat.completion.chunk",
                choices: [{ delta: {}, finish_reason: "stop" }],
              },
            ],
          }),
        )

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "hi")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const run = yield* handle
          .process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "hi" }],
            tools: {},
          })
          .pipe(Effect.forkChild)

        yield* waitFor(
          MessageV2.parts(msg.id).pipe(
            Effect.map((parts) => parts.find((part): part is SessionV1.TextPart => part.type === "text")),
            Effect.provideService(Database.Service, database),
          ),
          "timed out waiting for text part",
        )
        yield* Effect.sleep("20 millis")
        gate.resolve()

        const exit = yield* Fiber.await(run)
        const text = (yield* MessageV2.parts(msg.id)).find((part): part is SessionV1.TextPart => part.type === "text")

        expect(Exit.isSuccess(exit)).toBe(true)
        expect(text?.text).toBe("hello")
        expect(text?.time?.start).toBeDefined()
        expect(text?.time?.end).toBeDefined()
        if (!text?.time?.start || !text.time.end) return
        expect(text.time.start).toBeLessThan(text.time.end)
      }),
    { config: (url) => providerCfg(url) },
  ),
)

it.live("session.processor effect tests stop after token overflow requests compaction", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const database = yield* Database.Service
        const { processors, session, provider } = yield* boot()

        yield* llm.text("after", { usage: { input: 100, output: 0 } })

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "compact")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const base = yield* provider.getModel(ref.providerID, ref.modelID)
        const mdl = { ...base, limit: { context: 20, output: 10 } }
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const value = yield* handle.process({
          user: {
            id: parent.id,
            sessionID: chat.id,
            role: "user",
            time: parent.time,
            agent: parent.agent,
            model: { providerID: ref.providerID, modelID: ref.modelID },
          } satisfies SessionV1.User,
          sessionID: chat.id,
          model: mdl,
          agent: agent(),
          system: [],
          messages: [{ role: "user", content: "compact" }],
          tools: {},
        })

        const parts = yield* MessageV2.parts(msg.id)

        expect(value).toBe("compact")
        expect(parts.some((part) => part.type === "text" && part.text === "after")).toBe(true)
        expect(parts.some((part) => part.type === "step-finish")).toBe(true)
      }),
    { config: (url) => providerCfg(url) },
  ),
)

it.live("session.processor effect tests capture reasoning from http mock", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const database = yield* Database.Service
        const { processors, session, provider } = yield* boot()

        yield* llm.push(reply().reason("think").text("done").stop())

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "reason")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const value = yield* handle.process({
          user: {
            id: parent.id,
            sessionID: chat.id,
            role: "user",
            time: parent.time,
            agent: parent.agent,
            model: { providerID: ref.providerID, modelID: ref.modelID },
          } satisfies SessionV1.User,
          sessionID: chat.id,
          model: mdl,
          agent: agent(),
          system: [],
          messages: [{ role: "user", content: "reason" }],
          tools: {},
        })

        const parts = yield* MessageV2.parts(msg.id)
        const reasoning = parts.find((part): part is SessionV1.ReasoningPart => part.type === "reasoning")
        const text = parts.find((part): part is SessionV1.TextPart => part.type === "text")

        expect(value).toBe("continue")
        expect(yield* llm.calls).toBe(1)
        expect(reasoning?.text).toBe("think")
        expect(text?.text).toBe("done")
      }),
    { config: (url) => providerCfg(url) },
  ),
)

it.live("session.processor effect tests reset reasoning state across retries", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const { processors, session, provider } = yield* boot()

        yield* llm.push(reply().reason("one").reset(), reply().reason("two").stop())

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "reason")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const value = yield* handle.process({
          user: {
            id: parent.id,
            sessionID: chat.id,
            role: "user",
            time: parent.time,
            agent: parent.agent,
            model: { providerID: ref.providerID, modelID: ref.modelID },
          } satisfies SessionV1.User,
          sessionID: chat.id,
          model: mdl,
          agent: agent(),
          system: [],
          messages: [{ role: "user", content: "reason" }],
          tools: {},
        })

        const parts = yield* MessageV2.parts(msg.id)
        const reasoning = parts.filter((part): part is SessionV1.ReasoningPart => part.type === "reasoning")

        expect(value).toBe("continue")
        expect(yield* llm.calls).toBe(2)
        expect(reasoning.some((part) => part.text === "two")).toBe(true)
        expect(reasoning.some((part) => part.text === "onetwo")).toBe(false)
      }),
    { config: (url) => providerCfg(url) },
  ),
)

it.live("session.processor effect tests do not retry unknown json errors", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const { processors, session, provider } = yield* boot()

        yield* llm.error(400, { error: { message: "no_kv_space" } })

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "json")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const value = yield* handle.process({
          user: {
            id: parent.id,
            sessionID: chat.id,
            role: "user",
            time: parent.time,
            agent: parent.agent,
            model: { providerID: ref.providerID, modelID: ref.modelID },
          } satisfies SessionV1.User,
          sessionID: chat.id,
          model: mdl,
          agent: agent(),
          system: [],
          messages: [{ role: "user", content: "json" }],
          tools: {},
        })

        expect(value).toBe("stop")
        expect(yield* llm.calls).toBe(1)
        expect(handle.message.error?.name).toBe("APIError")
      }),
    { config: (url) => providerCfg(url) },
  ),
)

it.live("session.processor effect tests retry recognized structured json errors", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const { processors, session, provider } = yield* boot()

        yield* llm.error(429, { type: "error", error: { type: "too_many_requests" } })
        yield* llm.text("after")

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "retry json")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const value = yield* handle.process({
          user: {
            id: parent.id,
            sessionID: chat.id,
            role: "user",
            time: parent.time,
            agent: parent.agent,
            model: { providerID: ref.providerID, modelID: ref.modelID },
          } satisfies SessionV1.User,
          sessionID: chat.id,
          model: mdl,
          agent: agent(),
          system: [],
          messages: [{ role: "user", content: "retry json" }],
          tools: {},
        })

        const parts = yield* MessageV2.parts(msg.id)

        expect(value).toBe("continue")
        expect(yield* llm.calls).toBe(2)
        expect(parts.some((part) => part.type === "text" && part.text === "after")).toBe(true)
        expect(handle.message.error).toBeUndefined()
      }),
    { config: (url) => providerCfg(url) },
  ),
)

it.live("session.processor effect tests publish retry status updates", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const { processors, session, provider } = yield* boot()
        const events = yield* EventV2Bridge.Service

        yield* llm.error(503, { error: "boom" })
        yield* llm.text("")

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "retry")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const states: number[] = []
        const off = yield* events.listen((evt) => {
          if (evt.type !== SessionStatus.Event.Status.type) return Effect.void
          const data = evt.data as typeof SessionStatus.Event.Status.data.Type
          if (data.sessionID === chat.id && data.status.type === "retry") states.push(data.status.attempt)
          return Effect.void
        })
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const value = yield* handle.process({
          user: {
            id: parent.id,
            sessionID: chat.id,
            role: "user",
            time: parent.time,
            agent: parent.agent,
            model: { providerID: ref.providerID, modelID: ref.modelID },
          } satisfies SessionV1.User,
          sessionID: chat.id,
          model: mdl,
          agent: agent(),
          system: [],
          messages: [{ role: "user", content: "retry" }],
          tools: {},
        })

        yield* off

        expect(value).toBe("continue")
        expect(yield* llm.calls).toBe(2)
        expect(states).toStrictEqual([1])
      }),
    { config: (url) => providerCfg(url) },
  ),
)

it.live("session.processor effect tests compact on structured context overflow", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const { processors, session, provider } = yield* boot()

        yield* llm.error(400, { type: "error", error: { code: "context_length_exceeded" } })

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "compact json")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const value = yield* handle.process({
          user: {
            id: parent.id,
            sessionID: chat.id,
            role: "user",
            time: parent.time,
            agent: parent.agent,
            model: { providerID: ref.providerID, modelID: ref.modelID },
          } satisfies SessionV1.User,
          sessionID: chat.id,
          model: mdl,
          agent: agent(),
          system: [],
          messages: [{ role: "user", content: "compact json" }],
          tools: {},
        })

        expect(value).toBe("compact")
        expect(yield* llm.calls).toBe(1)
        expect(handle.message.error).toBeUndefined()
      }),
    { config: (url) => providerCfg(url) },
  ),
)

it.live("session.processor effect tests complete AI SDK tool calls when native flag is off", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const { processors, session, provider } = yield* boot()

        yield* llm.tool("lookup", { query: "weather" })

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "tool")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const value = yield* handle.process({
          user: {
            id: parent.id,
            sessionID: chat.id,
            role: "user",
            time: parent.time,
            agent: parent.agent,
            model: { providerID: ref.providerID, modelID: ref.modelID },
          } satisfies SessionV1.User,
          sessionID: chat.id,
          model: mdl,
          agent: agent(),
          system: [],
          messages: [{ role: "user", content: "tool" }],
          tools: {
            lookup: tool({
              description: "Look up information",
              inputSchema: z.object({ query: z.string() }),
              execute: async (input) => ({
                title: "Weather lookup",
                output: `result:${input.query}`,
                metadata: { source: "test" },
              }),
            }),
          },
        })

        const parts = yield* MessageV2.parts(msg.id)
        const call = parts.find((part): part is SessionV1.ToolPart => part.type === "tool")

        expect(value).toBe("continue")
        expect(yield* llm.calls).toBe(1)
        expect(call?.callID).toBe("call_1")
        expect(call?.tool).toBe("lookup")
        expect(call?.state.status).toBe("completed")
        if (call?.state.status !== "completed") return
        expect(call.state.input).toEqual({ query: "weather" })
        expect(call.state.output).toBe("result:weather")
        expect(call.state.title).toBe("Weather lookup")
        expect(call.state.metadata).toEqual({ source: "test" })
        expect(call.state.time.start).toBeDefined()
        expect(call.state.time.end).toBeDefined()
      }),
    { config: (url) => providerCfg(url) },
  ),
)

it.live("session.processor effect tests mark pending tools as aborted on cleanup", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const database = yield* Database.Service
        const { processors, session, provider } = yield* boot()

        yield* llm.toolHang("bash", { cmd: "pwd" })

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "tool abort")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const run = yield* handle
          .process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "tool abort" }],
            tools: {},
          })
          .pipe(Effect.forkChild)

        yield* llm.wait(1)
        yield* waitFor(
          MessageV2.parts(msg.id).pipe(
            Effect.map((parts) => parts.find((part): part is SessionV1.ToolPart => part.type === "tool")),
            Effect.provideService(Database.Service, database),
          ),
          "timed out waiting for tool part",
        )
        yield* Fiber.interrupt(run)

        const exit = yield* Fiber.await(run)
        const parts = yield* MessageV2.parts(msg.id)
        const call = parts.find((part): part is SessionV1.ToolPart => part.type === "tool")

        expect(Exit.isFailure(exit)).toBe(true)
        if (Exit.isFailure(exit)) {
          expect(Cause.hasInterruptsOnly(exit.cause)).toBe(true)
        }
        expect(yield* llm.calls).toBe(1)
        expect(call?.state.status).toBe("error")
        if (call?.state.status === "error") {
          expect(call.state.error).toBe("Tool execution aborted")
          expect(call.state.metadata?.interrupted).toBe(true)
          expect(call.state.time.end).toBeDefined()
        }
      }),
    { config: (url) => providerCfg(url) },
  ),
)

it.live("session.processor effect tests record aborted errors and idle state", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const seen = defer<void>()
        const { processors, session, provider } = yield* boot()
        const events = yield* EventV2Bridge.Service
        const sts = yield* SessionStatus.Service

        yield* llm.hang

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "abort")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const errs: string[] = []
        const off = yield* events.listen((evt) => {
          if (evt.type !== Session.Event.Error.type) return Effect.void
          const data = evt.data as typeof Session.Event.Error.data.Type
          if (data.sessionID !== chat.id || !data.error) return Effect.void
          errs.push(data.error.name)
          seen.resolve()
          return Effect.void
        })
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const run = yield* handle
          .process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "abort" }],
            tools: {},
          })
          .pipe(Effect.forkChild)

        yield* llm.wait(1)
        yield* Fiber.interrupt(run)

        const exit = yield* Fiber.await(run)
        yield* Effect.promise(() => seen.promise)
        const stored = yield* MessageV2.get({ sessionID: chat.id, messageID: msg.id })
        const state = yield* sts.get(chat.id)
        yield* off

        expect(Exit.isFailure(exit)).toBe(true)
        if (Exit.isFailure(exit)) {
          expect(Cause.hasInterruptsOnly(exit.cause)).toBe(true)
        }
        expect(handle.message.error?.name).toBe("MessageAbortedError")
        expect(stored.info.role).toBe("assistant")
        if (stored.info.role === "assistant") {
          expect(stored.info.error?.name).toBe("MessageAbortedError")
        }
        expect(state).toMatchObject({ type: "idle" })
        expect(errs).toContain("MessageAbortedError")
      }),
    { config: (url) => providerCfg(url) },
  ),
)

it.live("session.processor effect tests mark interruptions aborted without manual abort", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const { processors, session, provider } = yield* boot()
        const sts = yield* SessionStatus.Service

        yield* llm.hang

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "interrupt")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const run = yield* handle
          .process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "interrupt" }],
            tools: {},
          })
          .pipe(Effect.forkChild)

        yield* llm.wait(1)
        yield* Fiber.interrupt(run)

        const exit = yield* Fiber.await(run)
        const stored = yield* MessageV2.get({ sessionID: chat.id, messageID: msg.id })
        const state = yield* sts.get(chat.id)

        expect(Exit.isFailure(exit)).toBe(true)
        expect(handle.message.error?.name).toBe("MessageAbortedError")
        expect(stored.info.role).toBe("assistant")
        if (stored.info.role === "assistant") {
          expect(stored.info.error?.name).toBe("MessageAbortedError")
        }
        expect(state).toMatchObject({ type: "idle" })
      }),
    { config: (url) => providerCfg(url) },
  ),
)

itProviderError.live("session.processor effect tests fail provider-executed error results", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const { processors, session, provider } = yield* boot()
        const events = yield* EventV2Bridge.Service

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "provider tool error")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const seen: string[] = []
        const off = yield* events.listen((event) => {
          seen.push(event.type)
          return Effect.void
        })
        const handle = yield* processors.create({ assistantMessage: msg, sessionID: chat.id, model: mdl })

        yield* handle.process({
          user: {
            id: parent.id,
            sessionID: chat.id,
            role: "user",
            time: parent.time,
            agent: parent.agent,
            model: { providerID: ref.providerID, modelID: ref.modelID },
          } satisfies SessionV1.User,
          sessionID: chat.id,
          model: mdl,
          agent: agent(),
          system: [],
          messages: [{ role: "user", content: "provider tool error" }],
          tools: {},
        })
        yield* off

        const parts = yield* MessageV2.parts(msg.id)
        const call = parts.find((part): part is SessionV1.ToolPart => part.type === "tool")
        expect(call?.state.status).toBe("error")
        if (call?.state.status === "error") expect(call.state.error).toBe("provider boom")
        expect(seen).toContain(MessageV2.Event.PartUpdated.type)
        expect(seen).toContain(MessageV2.Event.Updated.type)
        expect(seen.filter((type) => type.startsWith("session.next."))).toEqual([])
      }),
    { config: cfg },
  ),
)

itFragmentFailure.live("session.processor effect tests retain partial legacy parts without v2 events", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const { processors, session, provider } = yield* boot()
        const events = yield* EventV2Bridge.Service

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "provider failure")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const seen: string[] = []
        const off = yield* events.listen((event) => {
          seen.push(event.type)
          return Effect.void
        })
        const handle = yield* processors.create({ assistantMessage: msg, sessionID: chat.id, model: mdl })

        expect(
          yield* handle.process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "provider failure" }],
            tools: {},
          }),
        ).toBe("stop")
        yield* off

        const parts = yield* MessageV2.parts(msg.id)
        expect(parts).toEqual(
          expect.arrayContaining([
            expect.objectContaining({ type: "text", text: "partial" }),
            expect.objectContaining({ type: "reasoning", text: "thinking" }),
          ]),
        )
        expect(seen).toContain(MessageV2.Event.PartUpdated.type)
        expect(seen).toContain(Session.Event.Error.type)
        expect(seen.filter((type) => type.startsWith("session.next."))).toEqual([])
      }),
    { config: cfg },
  ),
)
