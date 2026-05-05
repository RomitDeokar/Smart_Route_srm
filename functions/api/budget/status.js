/* GET /api/budget/status — return current budget state. */

import { jsonResponse } from "../_shared/auth.js";
import { _budgetGet } from "./_store.js";

export const onRequestOptions = () => jsonResponse({ ok: true });
export const onRequestGet = () => jsonResponse({ ok: true, budget: _budgetGet() });
