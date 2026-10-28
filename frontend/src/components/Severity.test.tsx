import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { SeverityChip, SeverityGlyph } from "./Severity";

describe("Severity presentation components", () => {
  it("renders SeverityGlyph with accessible SVG shapes", () => {
    const { container, rerender } = render(<SeverityGlyph label="High" />);
    // Triangle has path
    expect(container.querySelector("path")).toBeInTheDocument();

    rerender(<SeverityGlyph label="Medium" />);
    expect(container.querySelector("path")).toBeInTheDocument();

    rerender(<SeverityGlyph label="Low" />);
    expect(container.querySelector("circle")).toBeInTheDocument();
  });

  it("renders SeverityChip with redundant shape and text label", () => {
    const { rerender } = render(<SeverityChip label="High" score={0.88} />);
    expect(screen.getByText("HIGH")).toBeInTheDocument();
    expect(screen.getByText("0.9")).toBeInTheDocument();

    rerender(<SeverityChip label="Low" />);
    expect(screen.getByText("LOW")).toBeInTheDocument();
  });

  it("handles unknown or undefined tier gracefully", () => {
    render(<SeverityChip label={undefined as any} />);
    expect(screen.getByText("UNKNOWN")).toBeInTheDocument();
  });
});
