export class DirtyManager {
  private isDirty: boolean = true;
  private listeners: (() => void)[] = [];

  public markDirty(): void {
    this.isDirty = true;
    for (let i = 0; i < this.listeners.length; i++) {
      this.listeners[i]();
    }
  }

  public checkAndClear(): boolean {
    const wasDirty = this.isDirty;
    this.isDirty = false;
    return wasDirty;
  }

  public getDirty(): boolean {
    return this.isDirty;
  }

  public onDirty(cb: () => void): () => void {
    this.listeners.push(cb);
    return () => {
      this.listeners = this.listeners.filter((l) => l !== cb);
    };
  }
}

