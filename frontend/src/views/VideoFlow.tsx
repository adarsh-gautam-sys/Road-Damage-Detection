import { FileVideo, Route } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { JobStatus } from "../api/types";
import { ErrorNote, GoldButton, PlainButton } from "../components/States";
import { useConfig } from "../state/ConfigContext";

const JOB_KEY = "rdai.video-job";

interface Props {
  conf: number;
  onDone: () => void;
  onViewMap: () => void;
}

export function VideoFlow({ conf, onDone, onViewMap }: Props) {
  const cfg = useConfig();
  const videoInput = useRef<HTMLInputElement>(null);
  const trackInput = useRef<HTMLInputElement>(null);
  const [video, setVideo] = useState<File | null>(null);
  const [track, setTrack] = useState<File | null>(null);
  const [jobId, setJobId] = useState<number | null>(() => {
    const raw = localStorage.getItem(JOB_KEY);
    return raw ? Number(raw) : null;
  });
  const [job, setJob] = useState<JobStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const onDoneRef = useRef(onDone);
  onDoneRef.current = onDone;

  // Poll the job. Persisted id means it survives the app going to the background or a reload.
  useEffect(() => {
    if (jobId === null) return;
    let stopped = false;
    let timer: number | undefined;
    const poll = async () => {
      if (stopped) return;
      try {
        const s = await api.job(jobId);
        if (stopped) return;
        setJob(s);
        setError(null);
        if (s.status === "done") {
          localStorage.removeItem(JOB_KEY);
          onDoneRef.current();
          return;
        }
        if (s.status === "failed") {
          localStorage.removeItem(JOB_KEY);
          return;
        }
      } catch (e) {
        if (stopped) return;
        if (e instanceof Error && "status" in e && (e as { status: number }).status === 404) {
          localStorage.removeItem(JOB_KEY);
          setJobId(null);
          return;
        }
        setError(e instanceof Error ? e.message : "Lost contact with the server.");
      }
      timer = window.setTimeout(() => void poll(), cfg.client.job_poll_interval_ms);
    };
    void poll();
    return () => {
      stopped = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [jobId, cfg.client.job_poll_interval_ms]);

  const maxMb = cfg.limits.max_video_mb;
  const tooBig = video !== null && video.size > maxMb * 1024 * 1024;

  const submit = useCallback(async () => {
    if (!video || tooBig) return;
    setBusy(true);
    setError(null);
    try {
      const res = await api.detectVideo(video, track, conf);
      localStorage.setItem(JOB_KEY, String(res.job_id));
      setJob(null);
      setJobId(res.job_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed.");
    } finally {
      setBusy(false);
    }
  }, [video, track, tooBig, conf]);

  const reset = () => {
    localStorage.removeItem(JOB_KEY);
    setJobId(null);
    setJob(null);
    setVideo(null);
    setTrack(null);
    setError(null);
  };

  const running = jobId !== null && (job === null || job.status === "queued" || job.status === "running");

  if (jobId !== null) {
    const pct = Math.round((job?.progress ?? 0) * 100);
    return (
      <div className="plate flex flex-col gap-4 p-5">
        <h2 className="font-label text-lg font-semibold">
          {job?.status === "done" ? "Video analysed" : job?.status === "failed" ? "Video analysis failed" : "Analysing video"}
        </h2>
        {job?.status !== "failed" && (
          <div>
            <div
              role="progressbar"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={pct}
              aria-label="Video analysis progress"
              className="h-3 overflow-hidden rounded-full bg-[#0c0e10] ring-1 ring-etch-strong"
            >
              <div className="h-full bg-gold transition-[width] duration-500 ease-[var(--ease-out)]" style={{ width: `${pct}%` }} />
            </div>
            <p className="tnum mt-2 font-mono text-[13px] text-ink-2">
              {job?.status === "done" ? "100%" : `${pct}%`} · {job?.status ?? "queued"}
            </p>
          </div>
        )}
        {running && (
          <p className="text-sm text-ink-2">You can leave this screen. The job keeps running on the server and resumes here when you return.</p>
        )}
        {job?.status === "failed" && <ErrorNote message={job.error ?? "The server could not process this video."} />}
        {job?.status === "done" && (
          <p className="text-ink-2">
            {String((job.summary as { unique_detections?: number }).unique_detections ?? 0)} unique defects after tracking, grouped into{" "}
            {String((job.summary as { segments_total?: number }).segments_total ?? 0)} segments in total.
          </p>
        )}
        {error && <ErrorNote message={error} />}
        <div className="flex flex-wrap gap-3">
          {job?.status === "done" && <GoldButton onClick={onViewMap}>View on map</GoldButton>}
          <PlainButton onClick={reset}>{running ? "Cancel tracking" : "Analyse another video"}</PlainButton>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <input ref={videoInput} type="file" accept="video/mp4,video/quicktime,video/*" className="sr-only" tabIndex={-1} onChange={(e) => setVideo(e.target.files?.[0] ?? null)} />
      <input ref={trackInput} type="file" accept=".gpx,.csv,text/csv,application/gpx+xml" className="sr-only" tabIndex={-1} onChange={(e) => setTrack(e.target.files?.[0] ?? null)} />

      <div className="plate flex flex-col gap-4 p-5">
        <p className="text-ink-2">
          Upload dashcam footage (up to {maxMb} MB). Adding a GPX or CSV track gives each defect a location; without one, defects are saved
          unlocated.
        </p>
        <div className="flex flex-wrap items-center gap-3">
          <PlainButton onClick={() => videoInput.current?.click()} className="min-h-12">
            <FileVideo size={18} aria-hidden="true" /> {video ? "Change video" : "Choose video"}
          </PlainButton>
          <PlainButton onClick={() => trackInput.current?.click()} className="min-h-12">
            <Route size={18} aria-hidden="true" /> {track ? "Change track" : "Add GPS track"}
          </PlainButton>
        </div>
        {video && (
          <p className="tnum font-mono text-[13px]">
            {video.name} · {(video.size / 1024 / 1024).toFixed(1)} MB{track ? ` · ${track.name}` : ""}
          </p>
        )}
        {tooBig && <p className="text-sm text-sev-high-text">This file is over the {maxMb} MB limit.</p>}
      </div>
      <GoldButton onClick={() => void submit()} disabled={!video || tooBig || busy}>
        {busy ? "Uploading…" : "Analyse video"}
      </GoldButton>
      {error && <ErrorNote message={error} onRetry={() => void submit()} />}
    </div>
  );
}
