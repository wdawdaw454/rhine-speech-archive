import { speechArchives, speechColumns } from './speech-catalog.ts';

export interface ArchiveRecord {
  id: string;
  title: string;
  en: string;
  department: string;
  category: string;
  date: string;
  lead: string;
  clearance: string;
  abstract: string;
  findings: string[];
  source: string;
}

export const records: ArchiveRecord[] = speechArchives;
export const categories = ["全部档案", ...speechColumns];
export const archiveColumns = speechColumns;
const gcd = (a: number, b: number): number => b ? gcd(b, a % b) : a;
// Recenter the infinite array only by a period shared by every column.
export const archiveRowPeriod = archiveColumns.map(name => records.filter(r => r.category === name).length)
  .reduce((period, count) => period * count / gcd(period, count), 1);

export function columnFiles(lane: number) {
  return records
    .map((record, index) => ({ record, index }))
    .filter(({ record }) => record.category === archiveColumns[lane])
    .map(({ index }) => index);
}
export function fileLocation(index: number) {
  const lane = archiveColumns.indexOf(records[index].category);
  const row = 12 + columnFiles(lane).indexOf(index);
  return { lane, row, slot: lane * 32 + row };
}
export function fileAtSlot(slot: number) {
  const files = columnFiles(Math.floor(slot / 32));
  return files[Math.max(0, Math.min(files.length - 1, (slot % 32) - 12))];
}
