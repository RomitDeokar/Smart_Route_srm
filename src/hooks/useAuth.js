/* useAuth.js — JWT auth hook */
import { useState, useCallback } from "react";

const TOKEN_KEY = "sr_token";
const USER_KEY  = "sr_user";

export default function useAuth() {
  const [token, setToken] = useState(() => localStorage.getItem(TOKEN_KEY));
  const [user,  setUser]  = useState(() => {
    try { return JSON.parse(localStorage.getItem(USER_KEY)); } catch { return null; }
  });

  const login = useCallback((tokenVal, userVal) => {
    localStorage.setItem(TOKEN_KEY, tokenVal);
    localStorage.setItem(USER_KEY, JSON.stringify(userVal));
    setToken(tokenVal);
    setUser(userVal);
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
    setToken(null);
    setUser(null);
  }, []);

  /* Authenticated fetch wrapper */
  const authFetch = useCallback(async (url, options = {}) => {
    const headers = {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(options.headers || {})
    };
    return fetch(url, { ...options, headers });
  }, [token]);

  return { token, user, login, logout, authFetch, isAuthed: !!token };
}
