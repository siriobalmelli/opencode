import { describe, expect, test } from "bun:test"
import { Effect } from "effect"
import { CrossSpawnSpawner } from "@opencode-ai/core/cross-spawn-spawner"
import { Npm } from "@opencode-ai/core/npm"
import path from "path"
import { pathToFileURL } from "url"
import { Account } from "../../src/account/account"
import { Auth } from "../../src/auth"
import { RuntimeFlags } from "../../src/effect/runtime-flags"
import { Plugin } from "../../src/plugin/index"
import { triggerProviderFailure } from "../../src/plugin/index"
import type { ChatProviderFailureInput, Hooks } from "@opencode-ai/plugin"

import { TestInstance } from "../fixture/fixture"
import { testEffect } from "../lib/effect"
import { AccountTest } from "../fake/account"
import { AuthTest } from "../fake/auth"
import { NpmTest } from "../fake/npm"
import { ProviderV2 } from "@opencode-ai/core/provider"
import { ModelV2 } from "@opencode-ai/core/model"
import { AppNodeBuilder } from "@opencode-ai/core/effect/app-node-builder"
import { LayerNode } from "@opencode-ai/core/effect/layer-node"

const it = testEffect(
  AppNodeBuilder.build(LayerNode.group([Plugin.node, CrossSpawnSpawner.node]), [
    [Auth.node, AuthTest.empty],
    [Account.node, AccountTest.empty],
    [Npm.node, NpmTest.noop],
    [RuntimeFlags.node, RuntimeFlags.layer({ disableDefaultPlugins: true })],
  ]),
)
const systemHook = "experimental.chat.system.transform"

const providerFailureInput: ChatProviderFailureInput = {
  sessionID: "ses_test",
  userMessageID: "msg_test",
  agent: "build",
  model: { providerID: "test", modelID: "test-model", variant: "primary" },
  failure: "server",
  observedTools: { pending: 0, running: 0, interrupted: 0, errored: 0, completed: 1 },
}

function withProject<A, E, R>(source: string, self: Effect.Effect<A, E, R>) {
  return Effect.gen(function* () {
    const test = yield* TestInstance
    const file = path.join(test.directory, "plugin.ts")
    yield* Effect.all(
      [
        Effect.promise(() => Bun.write(file, source)),
        Effect.promise(() =>
          Bun.write(
            path.join(test.directory, "opencode.json"),
            JSON.stringify(
              {
                $schema: "https://opencode.ai/config.json",
                plugin: [pathToFileURL(file).href],
              },
              null,
              2,
            ),
          ),
        ),
      ],
      { discard: true, concurrency: 2 },
    )
    return yield* self
  })
}

const triggerSystemTransform = Effect.fn("PluginTriggerTest.triggerSystemTransform")(function* () {
  const plugin = yield* Plugin.Service
  const out = { system: [] as string[] }
  yield* plugin.trigger(
    systemHook,
    {
      model: {
        providerID: ProviderV2.ID.anthropic,
        modelID: ModelV2.ID.make("claude-sonnet-4-6"),
      },
    },
    out,
  )
  return out.system
})

const triggerFailure = Effect.fn("PluginTriggerTest.triggerFailure")(function* () {
  const plugin = yield* Plugin.Service
  return yield* plugin.triggerProviderFailure(providerFailureInput)
})

describe("plugin.trigger", () => {
  it.instance("runs synchronous hooks without crashing", () =>
    withProject(
      [
        "export default async () => ({",
        `  ${JSON.stringify(systemHook)}: (_input, output) => {`,
        '    output.system.unshift("sync")',
        "  },",
        "})",
        "",
      ].join("\n"),
      Effect.gen(function* () {
        expect(yield* triggerSystemTransform()).toEqual(["sync"])
      }),
    ),
  )

  it.instance("awaits asynchronous hooks", () =>
    withProject(
      [
        "export default async () => ({",
        `  ${JSON.stringify(systemHook)}: async (_input, output) => {`,
        "    await Bun.sleep(1)",
        '    output.system.unshift("async")',
        "  },",
        "})",
        "",
      ].join("\n"),
      Effect.gen(function* () {
        expect(yield* triggerSystemTransform()).toEqual(["async"])
      }),
    ),
  )
})

describe("plugin.triggerProviderFailure", () => {
  it.instance("converts thrown hooks to typed failures", () =>
    withProject(
      'export default async () => ({ "chat.provider.failure": () => { throw new Error("provider hook exploded") } })',
      Effect.gen(function* () {
        expect(yield* triggerFailure().pipe(Effect.flip)).toBe("provider hook exploded")
      }),
    ),
  )

  test("returns unhandled when no hook is registered", async () => {
    await expect(triggerProviderFailure([], providerFailureInput)).resolves.toEqual({ action: "unhandled", models: [] })
  })

  test("stops at the first handled stop", async () => {
    let laterCalled = false
    let received: ChatProviderFailureInput | undefined
    const hooks: Hooks[] = [
      {
        "chat.provider.failure": async (input, output) => {
          received = input
          output.action = "stop"
        },
      },
      {
        "chat.provider.failure": async () => {
          laterCalled = true
        },
      },
    ]

    await expect(triggerProviderFailure(hooks, providerFailureInput)).resolves.toEqual({ action: "stop", models: [] })
    expect(received?.observedTools).toEqual({ pending: 0, running: 0, interrupted: 0, errored: 0, completed: 1 })
    expect(laterCalled).toBe(false)
  })

  test("stops at the first handled fallback", async () => {
    let laterCalled = false
    const hooks: Hooks[] = [
      {
        "chat.provider.failure": async (_input, output) => {
          output.action = "fallback"
          output.models.push({ providerID: "fallback", modelID: "next", variant: "secondary" })
        },
      },
      {
        "chat.provider.failure": async () => {
          laterCalled = true
        },
      },
    ]

    await expect(triggerProviderFailure(hooks, providerFailureInput)).resolves.toEqual({
      action: "fallback",
      models: [{ providerID: "fallback", modelID: "next", variant: "secondary" }],
    })
    expect(laterCalled).toBe(false)
  })

  test("never falls back from an unknown failure", async () => {
    const hooks: Hooks[] = [
      {
        "chat.provider.failure": async (_input, output) => {
          output.action = "fallback"
          output.models.push({ providerID: "fallback", modelID: "next" })
        },
      },
      {
        "chat.provider.failure": async (_input, output) => {
          output.action = "stop"
        },
      },
    ]

    await expect(triggerProviderFailure(hooks, { ...providerFailureInput, failure: "unknown" })).resolves.toEqual({
      action: "stop",
      models: [],
    })
  })
})
