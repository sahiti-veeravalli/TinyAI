"use client";

import React, { useState } from "react";
import { Sparkles, X, Send, BookOpen, ExternalLink, Bot, User } from "lucide-react";
import { chatAssistant } from "@/lib/api";

interface Message {
  role: "user" | "assistant";
  content: string;
  citations?: { title: string; source: string }[];
}

interface AIAssistantProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function AIAssistant({ isOpen, onClose }: AIAssistantProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content:
        "Hello! I am the TinyAI Research Assistant. Ask me anything about our patentable domain-aware compression schedule, experimental results on Kaggle Credit Card Fraud, or INT8 dynamic quantization trade-offs.",
      citations: [
        { title: "Patent Claim 1: Domain-Aware Compression Scheduler", source: "Patent Disclosure Section 4" },
      ],
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSend = async (text?: string) => {
    const query = text || input;
    if (!query.trim() || loading) return;

    const userMsg: Message = { role: "user", content: query };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await chatAssistant([...messages, userMsg], "credit_card_fraud");
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: res.reply,
          citations: res.citations,
        },
      ]);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "That broke. Not our finest hour. Backend assistant offline.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-y-0 right-0 z-50 w-full sm:w-[480px] bg-surface/95 backdrop-blur-2xl border-l border-border shadow-2xl flex flex-col font-mono text-xs animate-in slide-in-from-right duration-300">
      {/* Header */}
      <div className="p-4 border-b border-border/80 flex items-center justify-between bg-surface-2/60">
        <div className="flex items-center space-x-2.5">
          <div className="p-1.5 rounded-lg bg-violet/20 border border-violet/40 text-violet">
            <Sparkles className="w-4 h-4" />
          </div>
          <div>
            <h3 className="font-display font-bold text-sm text-text">AI Research Assistant</h3>
            <span className="text-[10px] text-muted">// RAG Knowledge Proxy_</span>
          </div>
        </div>
        <button
          onClick={onClose}
          className="p-1.5 rounded-lg hover:bg-surface-2 text-muted hover:text-text transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Suggested Questions */}
      <div className="p-3 bg-surface-2/40 border-b border-border/60 flex flex-wrap gap-1.5">
        {[
          "Why did INT8 hurt recall on fraud?",
          "Explain the patentable core",
          "What is the fairness audit result?",
        ].map((q, idx) => (
          <button
            key={idx}
            onClick={() => handleSend(q)}
            className="text-[10px] px-2.5 py-1 rounded-full bg-surface border border-border hover:border-violet text-muted hover:text-text transition-colors"
          >
            {q}
          </button>
        ))}
      </div>

      {/* Messages Scroll Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((m, idx) => {
          const isUser = m.role === "user";
          return (
            <div
              key={idx}
              className={`flex items-start space-x-2.5 ${isUser ? "flex-row-reverse space-x-reverse" : ""}`}
            >
              <div
                className={`p-1.5 rounded-lg border flex-shrink-0 ${
                  isUser
                    ? "bg-lime/15 border-lime/40 text-lime"
                    : "bg-violet/15 border-violet/40 text-violet"
                }`}
              >
                {isUser ? <User className="w-3.5 h-3.5" /> : <Bot className="w-3.5 h-3.5" />}
              </div>

              <div
                className={`max-w-[85%] p-3.5 rounded-xl border text-xs leading-relaxed space-y-2.5 ${
                  isUser
                    ? "bg-lime/10 border-lime/30 text-text"
                    : "bg-surface-2/90 border-border text-text"
                }`}
              >
                <div className="whitespace-pre-wrap">{m.content}</div>

                {/* Citations block */}
                {m.citations && m.citations.length > 0 && (
                  <div className="pt-2 border-t border-border/60 text-[10px] text-muted space-y-1">
                    <span className="font-bold text-violet block uppercase">// CITED SOURCES:</span>
                    {m.citations.map((c, cIdx) => (
                      <div key={cIdx} className="flex items-center space-x-1.5">
                        <BookOpen className="w-3 h-3 text-muted" />
                        <span>
                          {c.title} ({c.source})
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          );
        })}

        {loading && (
          <div className="flex items-center space-x-2 text-xs text-muted font-mono animate-pulse">
            <Sparkles className="w-3.5 h-3.5 text-violet" />
            <span>Consulting RAG vector corpus...</span>
          </div>
        )}
      </div>

      {/* Input Box */}
      <div className="p-3 border-t border-border/80 bg-surface-2/50">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSend();
          }}
          className="flex items-center space-x-2"
        >
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about compression, proofs, or results..."
            className="flex-1 bg-surface border border-border rounded-lg px-3 py-2 text-xs text-text placeholder-muted/60 focus:outline-none focus:border-violet"
          />
          <button
            type="submit"
            disabled={!input.trim() || loading}
            className="p-2 rounded-lg bg-lime text-black hover:bg-lime/90 transition-colors disabled:opacity-40"
          >
            <Send className="w-4 h-4" />
          </button>
        </form>
        <span className="text-[9px] text-muted block mt-1 text-center">
          AI-generated research proxy. Prompt injection protected.
        </span>
      </div>
    </div>
  );
}
