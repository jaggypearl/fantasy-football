// A ?week= value from the URL, or null if missing or not a regular-season week.
export function parseWeek(value: string | null): number | null {
  const week = Number(value);
  return value && Number.isInteger(week) && week >= 1 && week <= 18 ? week : null;
}
