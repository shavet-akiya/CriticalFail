"use client";

import { useEffect, useState } from "react";

export default function Home() {
  const [value, setValue] = useState<number | "">("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Use same-origin proxy paths; Next.js rewrites /api/* to backend
  const baseUrl = "/api";

  async function fetchValue() {
    setLoading(true);
    setError(null);
    try {
  const res = await fetch(`${baseUrl}/number`, { cache: "no-store" });
      if (!res.ok) throw new Error(`GET failed: ${res.status}`);
      const data = (await res.json()) as { value: number };
      setValue(data.value);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  async function updateValue() {
    if (value === "") return;
    if (value < 1 || value > 10) {
      setError("Value must be between 1 and 10");
      return;
    }
    setLoading(true);
    setError(null);
    try {
  const res = await fetch(`${baseUrl}/number`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ value: Number(value) }),
      });
      if (!res.ok) throw new Error(`POST failed: ${res.status}`);
      const data = (await res.json()) as { value: number };
      setValue(data.value);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetchValue();
  }, []);

  return (
    <div className="flex h-screen items-center justify-center bg-base-200">
      <div className="card w-96 bg-base-100 shadow-xl">
      <div className="card-body">
        <h2 className="card-title">Update Number</h2>

        <input
        type="number"
        className="input input-bordered w-full"
        required
        placeholder="Type a number between 1 to 10"
        min={1}
        max={10}
        title="Must be between be 1 to 10"
        value={value}
        onChange={(e) =>
          setValue(e.target.value === "" ? "" : Number(e.target.value))
        }
        disabled={loading}
        />

        <p className="text-sm text-gray-500">Must be between be 1 to 10</p>

        {loading && <p className="text-sm">Loading…</p>}
        {error && (
        <p className="text-sm text-error" role="alert">
          {error}
        </p>
        )}

        <div className="card-actions justify-end">
        <button
          className="btn btn-primary rounded-xl"
          onClick={updateValue}
          disabled={loading || value === ""}
        >
          Update
        </button>
        <button className="btn btn-secondary rounded-xl" onClick={fetchValue} disabled={loading}>
          Refresh
        </button>
        </div>
      </div>
      </div>
    </div>
  );
}
