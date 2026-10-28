import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { LcdReading } from "./LcdReading";
import { LedgerLine, LedgerTable } from "./Ledger";
import type { PreviewDetection, PriorityTerms } from "../api/types";

describe("LcdReading component", () => {
  it("renders empty state placeholder when no detection selected", () => {
    render(<LcdReading det={null} empty="SCANNING RETICLE" />);
    expect(screen.getByText("SCANNING RETICLE")).toBeInTheDocument();
  });

  it("renders measured metrics, class code, and confidence when detection provided", () => {
    const mockDet: PreviewDetection = {
      class_id: 3,
      class_code: "D40",
      class_name: "Pothole",
      confidence: 0.92,
      box: [100, 150, 300, 400],
      severity: {
        score: 78.5,
        label: "High",
        extent: 0.042,
        measure: "area_ratio",
      },
    };
    render(<LcdReading det={mockDet} empty="SCANNING" />);
    expect(screen.getByText("D40")).toBeInTheDocument();
    expect(screen.getByText(/Pothole/i)).toBeInTheDocument();
    expect(screen.getByText("HIGH")).toBeInTheDocument();
    expect(screen.getByText(/0.042/)).toBeInTheDocument();
    expect(screen.getByText(/78.5/)).toBeInTheDocument();
  });
});

describe("Ledger components", () => {
  const mockWeights: PriorityTerms = {
    s_max: 0.4,
    density: 0.3,
    road_importance: 0.2,
    confidence: 0.1,
  };
  const mockTerms: Partial<PriorityTerms> = {
    s_max: 80,
    density: 40,
    road_importance: 70,
    confidence: 50,
  };
  const mockWeighted: Partial<PriorityTerms> = {
    s_max: 32.0,
    density: 12.0,
    road_importance: 14.0,
    confidence: 5.0,
  };

  it("renders LedgerLine with 4 terms and total sum", () => {
    render(<LedgerLine weighted={mockWeighted} total={63.0} />);
    expect(screen.getByText(/32\.0/)).toBeInTheDocument();
    expect(screen.getByText(/12\.0/)).toBeInTheDocument();
    expect(screen.getByText(/14\.0/)).toBeInTheDocument();
    expect(screen.getByText(/5\.0/)).toBeInTheDocument();
    expect(screen.getByText(/= 63\.0/)).toBeInTheDocument();
  });

  it("renders LedgerTable with correct column breakdown", () => {
    render(
      <LedgerTable
        terms={mockTerms}
        weighted={mockWeighted}
        weights={mockWeights}
        total={63.0}
      />
    );
    expect(screen.getByText("Worst severity")).toBeInTheDocument();
    expect(screen.getByText("0.40")).toBeInTheDocument();
    expect(screen.getByText("80.0")).toBeInTheDocument();
    expect(screen.getByText("32.0")).toBeInTheDocument();
    expect(screen.getByRole("rowheader", { name: "Priority" })).toBeInTheDocument();
    expect(screen.getByText("63.0")).toBeInTheDocument();
  });
});
