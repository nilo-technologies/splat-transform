const EMPTY = -1;

// Fibonacci / murmur-style mixing constants.
const MUL_LO = 0x9E3779B9;
const MUL_HI = 0x85EBCA6B;
const MUL_FIN = 0x2545F491;

/**
 * Open-addressing hash map from a non-negative integer key to an `Int32`
 * value, backed by typed arrays.
 *
 * Exists because `Map` and `Set` cannot hold the large keyed collections the
 * voxel and mesh pipelines need: V8 caps both at 2^24 entries and throws
 * `RangeError: Map maximum size exceeded` past that. A voxel-face mesh of a fine
 * grid needs tens of millions of unique vertices, and a fine voxel grid holds
 * more blocks than that. Typed-array storage also avoids the per-entry
 * allocation and GC pressure a collection of that size would incur.
 *
 * Keys are stored in a `Float64Array` rather than an `Int32Array` because they
 * routinely exceed 32 bits — vertex keys grow as `coordStride^2 * z`, and block
 * indices as `nbx * nby * nbz` — and doubles hold every integer up to 2^53
 * exactly. Keys must be non-negative, since `-1` is the empty-slot sentinel.
 *
 * Linear probing, no deletion.
 */
class IntKeyMap {
    /** Slot keys, `-1` where empty. */
    keys: Float64Array;

    /** Slot values, meaningful only where `keys` is not `-1`. */
    values: Int32Array;

    private _size: number;
    private _capacity: number;
    private _mask: number;

    constructor(initialCapacity = 4096) {
        const cap = 1 << (32 - Math.clz32(Math.max(15, initialCapacity - 1)));
        this._capacity = cap;
        this._mask = cap - 1;
        this._size = 0;
        this.keys = new Float64Array(cap).fill(EMPTY);
        this.values = new Int32Array(cap);
    }

    get size(): number {
        return this._size;
    }

    /**
     * Mix a key up to 2^53 down to a slot index.
     *
     * Both 32-bit halves are folded in, so keys differing only above bit 32
     * (adjacent Z planes of a large grid) do not collide.
     *
     * @param key - Non-negative integer key.
     * @returns Slot index in `[0, capacity)`.
     */
    private _hash(key: number): number {
        const lo = key | 0;                       // low 32 bits (ToInt32 wraps)
        const hi = (key / 0x100000000) | 0;       // bits 32 and above
        let h = (Math.imul(lo, MUL_LO) ^ Math.imul(hi, MUL_HI)) >>> 0;
        h ^= h >>> 15;
        h = Math.imul(h, MUL_FIN) >>> 0;
        h ^= h >>> 13;
        return (h >>> 0) & this._mask;
    }

    /**
     * Find the slot a key occupies, or the empty slot it would be inserted at.
     *
     * @param key - Non-negative integer key.
     * @returns Slot index; `keys[slot]` is either `key` or `-1`.
     */
    slot(key: number): number {
        const mask = this._mask;
        const keys = this.keys;
        let i = this._hash(key);
        while (true) {
            const k = keys[i];
            if (k === key || k === EMPTY) return i;
            i = (i + 1) & mask;
        }
    }

    /**
     * Insert at a slot already known to be empty.
     *
     * May grow the table, which invalidates every previously returned slot
     * index, so callers must not reuse a slot after calling this.
     *
     * @param slot - Empty slot index obtained from `slot()`.
     * @param key - Non-negative integer key.
     * @param value - Value to store.
     */
    insertAt(slot: number, key: number, value: number): void {
        this.keys[slot] = key;
        this.values[slot] = value;
        this._size++;
        if (this._size > ((this._capacity * 0.7) | 0)) {
            this._grow();
        }
    }

    /**
     * Look up a key.
     *
     * @param key - Non-negative integer key.
     * @returns Stored value, or `-1` when the key is absent.
     */
    get(key: number): number {
        const s = this.slot(key);
        return this.keys[s] === EMPTY ? -1 : this.values[s];
    }

    /**
     * Test whether a key is present.
     *
     * @param key - Non-negative integer key.
     * @returns True when the key has a stored value.
     */
    has(key: number): boolean {
        return this.keys[this.slot(key)] !== EMPTY;
    }

    /**
     * Insert or overwrite a key.
     *
     * @param key - Non-negative integer key.
     * @param value - Value to store.
     */
    set(key: number, value: number): void {
        const s = this.slot(key);
        if (this.keys[s] === EMPTY) {
            this.insertAt(s, key, value);
        } else {
            this.values[s] = value;
        }
    }

    /**
     * Release the backing storage. The map must not be used afterwards.
     */
    releaseStorage(): void {
        this.keys = new Float64Array(0);
        this.values = new Int32Array(0);
        this._size = 0;
        this._capacity = 0;
        this._mask = 0;
    }

    private _grow(): void {
        const oldKeys = this.keys;
        const oldValues = this.values;
        const oldCap = this._capacity;

        this._capacity *= 2;
        this._mask = this._capacity - 1;
        this.keys = new Float64Array(this._capacity).fill(EMPTY);
        this.values = new Int32Array(this._capacity);

        for (let i = 0; i < oldCap; i++) {
            const k = oldKeys[i];
            if (k !== EMPTY) {
                const s = this.slot(k);
                this.keys[s] = k;
                this.values[s] = oldValues[i];
            }
        }
    }
}

export { IntKeyMap };
