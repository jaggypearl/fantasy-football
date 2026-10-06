"use client";

interface Props {
  weeks: number[] | null; // null while loading or unavailable
  value: number | null;
  onChange: (week: number) => void;
}

export default function WeekSelect({ weeks, value, onChange }: Props) {
  const options = weeks ?? (value ? [value] : []);
  return (
    <>
      <label htmlFor="week-select" className="visually-hidden">
        Week
      </label>
      <select
        id="week-select"
        className="select"
        value={value ?? ""}
        disabled={!weeks}
        onChange={(e) => onChange(Number(e.target.value))}
      >
        {options.length === 0 && <option value="">Week</option>}
        {options.map((week) => (
          <option key={week} value={week}>
            Week {week}
          </option>
        ))}
      </select>
    </>
  );
}
