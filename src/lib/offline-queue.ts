import type { PatientDetails, ScreeningResult } from './screenings';

const DATABASE = 'retinasathi-offline-v1';
const STORE = 'pending-screenings';
const RETENTION_MS = 7 * 24 * 60 * 60 * 1000;
const synchronizationLocks = new Map<string, Promise<SyncResult>>();

type SyncResult = { synchronized: number; remaining: number };

export type PendingScreening = {
  id: string;
  userId: string;
  patient: PatientDetails;
  file: File;
  result: ScreeningResult;
  createdAt: string;
};

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE, 1);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(STORE)) database.createObjectStore(STORE, { keyPath: 'id' });
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error('Offline storage is unavailable.'));
  });
}

export async function queueScreening(userId: string, patient: PatientDetails, file: File, result: ScreeningResult, operationId = crypto.randomUUID()): Promise<void> {
  const database = await openDatabase();
  await new Promise<void>((resolve, reject) => {
    const transaction = database.transaction(STORE, 'readwrite');
    transaction.objectStore(STORE).put({ id: operationId, userId, patient, file, result, createdAt: new Date().toISOString() } satisfies PendingScreening);
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error ?? new Error('Could not queue the screening.'));
  });
  database.close();
}

export async function pendingScreenings(userId: string): Promise<PendingScreening[]> {
  const database = await openDatabase();
  const rows = await new Promise<PendingScreening[]>((resolve, reject) => {
    const transaction = database.transaction(STORE, 'readwrite');
    const store = transaction.objectStore(STORE);
    const request = store.getAll();
    request.onsuccess = () => {
      const now = Date.now();
      const retained = (request.result as PendingScreening[]).filter((row) => {
        const created = Date.parse(row.createdAt);
        const expired = !Number.isFinite(created) || now - created > RETENTION_MS;
        if (expired) store.delete(row.id);
        return !expired && row.userId === userId;
      });
      transaction.oncomplete = () => resolve(retained);
    };
    request.onerror = () => reject(request.error ?? new Error('Could not read the offline queue.'));
    transaction.onerror = () => reject(transaction.error ?? new Error('Could not maintain the offline queue.'));
  });
  database.close();
  return rows.sort((left, right) => left.createdAt.localeCompare(right.createdAt));
}

export async function clearPendingScreenings(userId: string): Promise<number> {
  const rows = await pendingScreenings(userId);
  for (const row of rows) await removePending(row.id);
  return rows.length;
}

async function removePending(id: string): Promise<void> {
  const database = await openDatabase();
  await new Promise<void>((resolve, reject) => {
    const transaction = database.transaction(STORE, 'readwrite');
    transaction.objectStore(STORE).delete(id);
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error ?? new Error('Could not clear a synchronized screening.'));
  });
  database.close();
}

export async function syncPendingScreenings(
  userId: string,
  save: (userId: string, patient: PatientDetails, file: File, result: ScreeningResult, operationId: string) => Promise<unknown>,
): Promise<SyncResult> {
  const synchronize = async (): Promise<SyncResult> => {
    const rows = await pendingScreenings(userId);
    let synchronized = 0;
    for (const row of rows) {
      try {
        await save(userId, row.patient, row.file, row.result, row.id);
        await removePending(row.id);
        synchronized += 1;
      } catch {
        break;
      }
    }
    return { synchronized, remaining: rows.length - synchronized };
  };

  if (typeof navigator !== 'undefined' && navigator.locks) {
    return navigator.locks.request(`retinasathi-offline-sync:${userId}`, synchronize);
  }

  const existing = synchronizationLocks.get(userId);
  if (existing) return existing;
  const task = synchronize().finally(() => {
    if (synchronizationLocks.get(userId) === task) synchronizationLocks.delete(userId);
  });
  synchronizationLocks.set(userId, task);
  return task;
}
