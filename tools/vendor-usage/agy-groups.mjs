// Stable AGY group identities and their public display names.  Consumers use
// the ID for selection and the displayName only for presentation.

export const AGY_GROUPS = Object.freeze([
  Object.freeze({ id: "gemini", displayName: "Gemini Models", buckets: Object.freeze({ five: "gemini-5h", seven: "gemini-weekly" }) }),
  Object.freeze({ id: "claude_gpt", displayName: "Claude and GPT models", buckets: Object.freeze({ five: "3p-5h", seven: "3p-weekly" }) }),
]);

export const AGY_GROUP_NAMES = Object.freeze(Object.fromEntries(
  AGY_GROUPS.map(({ id, displayName }) => [id, displayName]),
));
