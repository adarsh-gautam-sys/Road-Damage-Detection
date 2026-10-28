import type { SegmentStatus } from "../api/types";

/** Mirrors ALLOWED_STATUS_TRANSITIONS in src/rdai/db/models.py (the API enforces it with a 409). */
export const NEXT_STATUS: Record<SegmentStatus, SegmentStatus[]> = {
  new: ["verified", "rejected"],
  verified: ["scheduled", "rejected"],
  scheduled: ["fixed"],
  fixed: [],
  rejected: [],
};

export const STATUS_LABEL: Record<SegmentStatus, string> = {
  new: "New",
  verified: "Verified",
  scheduled: "Scheduled",
  fixed: "Fixed",
  rejected: "Rejected",
};

export const STATUS_ACTION: Record<SegmentStatus, string> = {
  new: "Reset",
  verified: "Verify",
  scheduled: "Schedule repair",
  fixed: "Mark fixed",
  rejected: "Reject",
};

export const ALL_STATUSES: SegmentStatus[] = ["new", "verified", "scheduled", "fixed", "rejected"];
