import type { ModelTuple } from "@opencode-ai/plugin"

export function validateFallbackCandidates(input: ModelTuple, candidates: unknown): ModelTuple[] | undefined {
  if (!Array.isArray(candidates) || candidates.length === 0) return undefined
  const seen = new Set<string>()
  const valid = candidates.map((candidate) => {
    if (!candidate || typeof candidate !== "object") return undefined
    const value = candidate as Record<string, unknown>
    if (typeof value.providerID !== "string" || value.providerID.length === 0) return undefined
    if (typeof value.modelID !== "string" || value.modelID.length === 0) return undefined
    if (value.variant !== undefined && (typeof value.variant !== "string" || value.variant.length === 0))
      return undefined
    const next = {
      providerID: value.providerID,
      modelID: value.modelID,
      ...(typeof value.variant === "string" ? { variant: value.variant } : {}),
    }
    const key = candidateKey(next)
    if (key === candidateKey(input) || seen.has(key)) return undefined
    seen.add(key)
    return next
  })
  return valid.every((candidate): candidate is ModelTuple => candidate !== undefined) ? valid : undefined
}

function candidateKey(candidate: ModelTuple) {
  return JSON.stringify([candidate.providerID, candidate.modelID, candidate.variant ?? null])
}

export * as SessionRouting from "./routing"
