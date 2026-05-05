/* POST /api/chat — AI travel assistant. Returns a structured mock reply
   that mirrors the GitHub project's chat schema. */

import { jsonResponse } from "./_shared/auth.js";

export const onRequestOptions = () => jsonResponse({ ok: true });

export const onRequestPost = async ({ request }) => {
  let body = {};
  try { body = await request.json(); } catch { body = {}; }
  const { message = "", context = {} } = body || {};
  const m = String(message).toLowerCase();

  const dest = context.destination || "your destination";
  const persona = context.persona || "explorer";
  const days = context.days || 5;
  const budget = context.budget || 18000;

  let reply, intent, quickActions, cards;

  if (/hotel|stay|accommodation|hostel|resort|lodge/.test(m)) {
    intent = "hotels";
    reply = `For ${dest} I'd recommend a 4-star property close to the city center. SRMIST students get priority for the SRM Hotel and on-campus hostels — apply via the SRMIST hostels portal for the best rates.`;
    quickActions = ["Show hotels in " + dest, "Find SRM-specific hostels", "Compare prices"];
    cards = [{ type: "hotels", title: `Top stays in ${dest}`,
               items: ["Lemon Tree Premier","Treebo Trend","SRM Hotel (if SRM city)","Zostel (hostel)"] }];
  } else if (/flight|train|cab|uber|ola|transport/.test(m)) {
    intent = "transport";
    reply = `Most trips to ${dest} have IndiGo/Air India direct flights from major Indian metros. Book 2–3 weeks ahead — fares start around ₹${Math.round(budget * 0.22).toLocaleString("en-IN")}.`;
    quickActions = ["Search flights", "Cab options", "Train alternatives"];
    cards = [{ type: "flights", title: "Flight options", items: ["IndiGo 6E","Air India AI","Vistara UK"] }];
  } else if (/food|eat|restaurant|cuisine/.test(m)) {
    intent = "food";
    reply = `${dest} has authentic local cuisine — your ${persona} persona suggests budget-aware food stops. I've allocated about ₹${Math.round(budget * 0.22).toLocaleString("en-IN")} for meals across ${days} days.`;
    quickActions = ["Top restaurants", "Local street food", "Vegetarian options"];
    cards = [{ type: "food", title: "Must-try in " + dest, items: ["Local thali","Heritage café","Street snacks"] }];
  } else if (/budget|cost|price|expense/.test(m)) {
    intent = "budget";
    reply = `For ₹${Number(budget).toLocaleString("en-IN")} over ${days} days, the Q-Learning optimizer suggests 35% stay, 22% food, 18% activities — that's ₹${Math.round(budget/days).toLocaleString("en-IN")}/day.`;
    quickActions = ["Optimize budget", "See breakdown", "Cheaper alternatives"];
    cards = [{ type: "budget", title: "Budget breakdown",
               items: [`Stay ₹${Math.round(budget*0.35).toLocaleString("en-IN")}`,
                       `Food ₹${Math.round(budget*0.22).toLocaleString("en-IN")}`,
                       `Activities ₹${Math.round(budget*0.18).toLocaleString("en-IN")}`] }];
  } else if (/weather|rain|forecast|climate/.test(m)) {
    intent = "weather";
    reply = `Weather forecast for ${dest} is being analyzed. Naive Bayes classifier suggests packing layers + rain backup if precipitation > 50%.`;
    quickActions = ["See forecast", "Packing list", "Indoor backups"];
    cards = [{ type: "weather", title: "Forecast", items: ["☀️ Sunny start","🌤️ Mild afternoon","🌧️ Possible evening shower"] }];
  } else if (/srm|hostel|campus|kattankulathur/.test(m)) {
    intent = "srm";
    reply = `For SRMIST students, on-campus hostels are available with WiFi, mess and 24x7 security. Apply via the SRMIST hostels portal. The SRM Hotel (Maamallan) is the official 4-star option for parents and visiting faculty.`;
    quickActions = ["Apply for hostel", "SRM Hotel info", "Near-campus hotels"];
    cards = [{ type: "srm", title: "SRM accommodations",
               items: ["Premium Boys/Girls Hostel","SRM Hotel Maamallan","GRT Grand Days (near campus)"] }];
  } else {
    intent = "general";
    reply = `I'm SmartRoute's AI assistant. I can help with hotels, flights, food, budget, weather and SRM-specific accommodations for ${dest}.`;
    quickActions = ["Find hotels", "Best food spots", "Optimize budget"];
    cards = [{ type: "info", title: "What I can help with",
               items: ["Multi-agent trip planning","Real hotel prices","SRM hostel applications","Budget optimization"] }];
  }

  return jsonResponse({ ok: true, reply, intent, quickActions, cards });
};
