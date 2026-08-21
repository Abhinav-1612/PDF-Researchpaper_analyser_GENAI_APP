/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      colors: {
        navy: {
          950: "#030712",
          900: "#060d1f",
          800: "#0a1628",
          700: "#0e1f35",
          600: "#132840",
        },
        cyan: {
          neon: "#00ffcc",
          glow: "#00e5b7",
        },
        purple: {
          neon: "#bf5fff",
          glow: "#9b33ff",
        },
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "float": "float 6s ease-in-out infinite",
        "glow": "glow 2s ease-in-out infinite alternate",
        "typing": "typing 1.2s steps(40) infinite",
        "spin-slow": "spin 8s linear infinite",
      },
      keyframes: {
        float: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-20px)" },
        },
        glow: {
          from: { boxShadow: "0 0 10px #00ffcc44, 0 0 20px #00ffcc22" },
          to:   { boxShadow: "0 0 20px #00ffcc88, 0 0 40px #00ffcc44" },
        },
      },
      backdropBlur: { xs: "2px" },
    },
  },
  plugins: [],
};
