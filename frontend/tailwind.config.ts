import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class"],
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: "var(--bg)",
        surface: "var(--surface)",
        "surface-2": "var(--surface-2)",
        border: "var(--border)",
        lime: {
          DEFAULT: "var(--lime)",
          glow: "rgba(198, 255, 61, 0.25)",
        },
        magenta: {
          DEFAULT: "var(--magenta)",
          glow: "rgba(255, 46, 147, 0.25)",
        },
        violet: {
          DEFAULT: "var(--violet)",
          glow: "rgba(123, 92, 255, 0.25)",
        },
        amber: {
          DEFAULT: "var(--amber)",
          glow: "rgba(255, 176, 32, 0.25)",
        },
        text: "var(--text)",
        muted: "var(--muted)",
      },
      fontFamily: {
        sans: ["var(--font-inter)", "sans-serif"],
        display: ["var(--font-space-grotesk)", "sans-serif"],
        mono: ["var(--font-jetbrains-mono)", "monospace"],
      },
      boxShadow: {
        neo: "4px 4px 0px 0px var(--magenta)",
        "neo-hover": "2px 2px 0px 0px var(--magenta)",
        "lime-glow": "0 0 25px -5px rgba(198, 255, 61, 0.4)",
        "magenta-glow": "0 0 25px -5px rgba(255, 46, 147, 0.4)",
        "violet-glow": "0 0 25px -5px rgba(123, 92, 255, 0.4)",
      },
      borderRadius: {
        DEFAULT: "10px",
        card: "12px",
      },
      animation: {
        pulseSlow: "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        scanline: "scanline 8s linear infinite",
        glitch: "glitch 0.2s ease-in-out infinite",
      },
      keyframes: {
        scanline: {
          "0%": { transform: "translateY(-100%)" },
          "100%": { transform: "translateY(1000%)" },
        },
        glitch: {
          "0%, 100%": { transform: "translate(0)" },
          "25%": { transform: "translate(-2px, 1px)" },
          "75%": { transform: "translate(2px, -1px)" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
