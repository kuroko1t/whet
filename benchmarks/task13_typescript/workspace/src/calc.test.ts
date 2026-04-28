import { test } from "node:test";
import assert from "node:assert/strict";
import { add, multiply } from "./calc.ts";

test("add returns the sum", () => {
    assert.equal(add(2, 3), 5);
    assert.equal(add(-1, 1), 0);
});

test("multiply returns the product", () => {
    assert.equal(multiply(3, 4), 12);
    assert.equal(multiply(0, 5), 0);
});
