"use client";

import React, { useState, useEffect } from "react";
import "./globals.css";

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const [terminalMode, setTerminalMode] = useState(false);

  // Konami Code Easter Egg: ↑ ↑ ↓ ↓ ← → ← → B A
  useEffect(() => {
    const konamiSequence = [
      "ArrowUp", "ArrowUp", "ArrowDown", "ArrowDown",
      "ArrowLeft", "ArrowRight", "ArrowLeft", "ArrowRight",
      "b", "a"
    ];
    let keyIndex = 0;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() === konamiSequence[keyIndex].toLowerCase()) {
        keyIndex++;
        if (keyIndex === konamiSequence.length) {
          setTerminalMode((prev) => !prev);
          keyIndex = 0;
        }
      } else {
        keyIndex = 0;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  return (
    <html lang="en" className={terminalMode ? "terminal-mode" : ""}>
      <head>
        <title>TinyAI — Edge AI Model Compression & Inference Platform</title>
        <meta
          name="description"
          content="A domain-aware model compression pipeline that brings high-accuracy AI to enterprise edge predictive endpoints while keeping accuracy within 1-2%."
        />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </head>
      <body className="bg-bg text-text antialiased selection:bg-magenta selection:text-white">
        {terminalMode && (
          <div className="fixed top-2 right-2 z-50 bg-black/80 border border-lime text-lime px-2 py-1 text-[10px] font-mono">
            [TERMINAL CRT MODE ACTIVE - KONAMI TRIGGERED]
          </div>
        )}
        {children}
      </body>
    </html>
  );
}
