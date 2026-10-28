import "@testing-library/jest-dom/vitest";

// Polyfill ResizeObserver for JSDOM
if (typeof window !== "undefined" && !window.ResizeObserver) {
  window.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  } as any;
}

// Mock Canvas 2D context for JSDOM
if (typeof HTMLCanvasElement !== "undefined") {
  HTMLCanvasElement.prototype.getContext = function (contextType: string) {
    if (contextType === "2d") {
      return {
        clearRect: () => {},
        fillRect: () => {},
        strokeRect: () => {},
        beginPath: () => {},
        closePath: () => {},
        moveTo: () => {},
        lineTo: () => {},
        stroke: () => {},
        fill: () => {},
        fillText: () => {},
        measureText: () => ({ width: 0 }),
        drawImage: () => {},
        setLineDash: () => {},
        setTransform: () => {},
        scale: () => {},
        translate: () => {},
        rotate: () => {},
        save: () => {},
        restore: () => {},
      } as any;
    }
    return null;
  } as any;
}
