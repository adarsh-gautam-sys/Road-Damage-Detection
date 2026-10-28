import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { api, type ClientConfig } from "./api/client";

const mockConfig: ClientConfig = {
  default_conf: 0.25,
  model: {
    name: "YOLO11n-RDD-India",
    run: "E1",
  },
  limits: {
    max_photo_mb: 15,
    max_video_mb: 100,
  },
  client: {
    preview_fps: 4,
    preview_max_edge_px: 640,
    preview_jpeg_quality: 0.7,
    upload_max_edge_px: 1280,
    upload_jpeg_quality: 0.85,
    auto_patrol_min_distance_m: 10,
    auto_patrol_min_interval_s: 5,
    gps_good_accuracy_m: 15,
    job_poll_interval_ms: 1500,
  },
  priority: {
    weights: { s_max: 0.4, density: 0.3, road_importance: 0.2, confidence: 0.1 },
    bands: { low_max: 35, medium_max: 65 },
    eps_meters: 50,
    density_saturation_count: 5,
  },
  severity: {
    tier_bounds: { low_upper: 35, medium_upper: 65, max_score: 100 },
    classes: {
      D40: { name: "Pothole", measure: "relative_area", e_low: 0.005, e_high: 0.08, min_label: "Low" },
      D20: { name: "Alligator crack", measure: "relative_area", e_low: 0.01, e_high: 0.1, min_label: "Low" },
      D00: { name: "Longitudinal crack", measure: "aspect_ratio", e_low: 3, e_high: 10, min_label: "Low" },
      D10: { name: "Transverse crack", measure: "aspect_ratio", e_low: 3, e_high: 10, min_label: "Low" },
    },
  },
};

describe("App shell & navigation", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(api, "config").mockResolvedValue(mockConfig);
    vi.spyOn(api, "health").mockResolvedValue({
      status: "healthy",
      model_loaded: true,
      version: "0.1.0",
      model_name: "YOLO11n-RDD-India",
      run: "E1",
    });
    vi.spyOn(api, "segments").mockResolvedValue({ total: 0, items: [] });
    vi.spyOn(api, "metrics").mockResolvedValue({
      status: "populated",
      generated_at: "2026-10-28T09:30:19.562937+00:00",
      optimal_conf: 0.193,
      experiments: {
        E1: {
          experiment_id: "E1",
          split: "test",
          params_m: 9.4,
          gpu_ms: 1.28,
          cpu_ms: 151.71,
          optimal_val_conf: 0.193,
          optimal_val_f1: 0.34,
          overall: { precision: 0.4487, recall: 0.3467, map50: 0.3655, map50_95: 0.1728 },
          per_class: {
            D00: { precision: 0.32, recall: 0.22, ap50: 0.24, ap50_95: 0.12 },
            D10: { precision: 0.43, recall: 0.15, ap50: 0.22, ap50_95: 0.10 },
            D20: { precision: 0.54, recall: 0.60, ap50: 0.58, ap50_95: 0.29 },
            D40: { precision: 0.49, recall: 0.39, ap50: 0.41, ap50_95: 0.17 },
          },
          confusion_matrix: [
            [77, 1, 2, 1, 202],
            [0, 2, 0, 0, 7],
            [6, 0, 201, 3, 203],
            [4, 0, 4, 230, 316],
            [157, 10, 86, 253, 0],
          ],
          metric_gate: {
            passed: false,
            map50: 0.3655,
            min_map50: 0.5,
            d40_ap50: 0.4105,
            min_pothole_ap50: 0.55,
          },
        },
      },
    } as any);
  });

  it("renders boot connecting state and loads configuration successfully", async () => {
    render(<App />);

    await waitFor(() => {
      expect(screen.getByText(/Road Damage AI/i)).toBeInTheDocument();
    });

    expect(screen.getByText(/YOLO11n-RDD-India/i)).toBeInTheDocument();
    expect(screen.getByText("Online")).toBeInTheDocument();
  });

  it("switches between views and navigates to Model Audit", async () => {
    const user = userEvent.setup();
    render(<App />);

    await waitFor(() => {
      expect(screen.getByText(/Road Damage AI/i)).toBeInTheDocument();
    });

    // Navigate to Model Audit
    const auditButtons = screen.getAllByRole("button", { name: /Model/i });
    await user.click(auditButtons[0]);

    await waitFor(() => {
      expect(screen.getByText("Model and method")).toBeInTheDocument();
      expect(screen.getByText(/reports\/metrics\.json/i)).toBeInTheDocument();
      expect(screen.getAllByText(/0\.365/).length).toBeGreaterThan(0);
    });
  });
});
