import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { api, type ClientConfig } from "../api/client";

interface ConfigState {
  config: ClientConfig | null;
  error: string | null;
  reload: () => void;
}

const Ctx = createContext<ConfigState>({ config: null, error: null, reload: () => undefined });

export function ConfigProvider({ children }: { children: ReactNode }) {
  const [config, setConfig] = useState<ClientConfig | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    let live = true;
    api
      .config()
      .then((c) => {
        if (live) {
          setConfig(c);
          setError(null);
        }
      })
      .catch((e: unknown) => {
        if (live) setError(e instanceof Error ? e.message : "Unknown error");
      });
    return () => {
      live = false;
    };
  }, [tick]);

  const reload = useCallback(() => {
    setError(null);
    setTick((t) => t + 1);
  }, []);

  return <Ctx.Provider value={{ config, error, reload }}>{children}</Ctx.Provider>;
}

export function useConfigState(): ConfigState {
  return useContext(Ctx);
}

/** Config that is guaranteed present: only call under the boot gate in App. */
export function useConfig(): ClientConfig {
  const { config } = useContext(Ctx);
  if (!config) throw new Error("useConfig called before config loaded");
  return config;
}
