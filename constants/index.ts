// constants/index.ts

export const BACKEND_URL = 
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8080";

export const LOCAL_URL = "http://localhost:3000";

// List of available VRM avatars in public/models/
export const AVATAR_LIST = [
  "Fox Loli",
  "Girl Next Door",
  "Elf",
  "Cute Girl",
  "Cool Guy",
  "Business Woman",
];

// Some avatars need to be flipped on Y axis
export const AVATAR_LIST_FLIP = [
  "Fox Loli",
  "Girl Next Door",
];

// Available Google Gemini TTS voices
export const GOOGLE_VOICE_LIST = [
  { voice_name: "Puck", description: "Male, friendly (English)" },
  { voice_name: "Charon", description: "Male, deep (English)" },
  { voice_name: "Kore", description: "Female, friendly (English)" },
  { voice_name: "Fenrir", description: "Male, energetic (English)" },
  { voice_name: "Aoede", description: "Female, warm (English)" },
  { voice_name: "Zephyr", description: "Female, soft (English)" },
];
