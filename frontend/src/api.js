const API_BASE = import.meta.env.VITE_API_BASE ?? "";

function buildUrl(path) {
  if (!API_BASE) {
    return path;
  }
  return `${API_BASE}${path}`;
}

async function parseError(response, fallbackMessage) {
  let message = fallbackMessage;

  try {
    const data = await response.json();
    message = data?.detail || data?.message || fallbackMessage;
  } catch {
    message = fallbackMessage;
  }

  return message;
}

export async function requestCspace(payload) {
  const response = await fetch(buildUrl("/api/cspace"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(await parseError(response, `C-space request failed with status ${response.status}`));
  }

  return response.json();
}

export async function requestPlan(payload) {
  const response = await fetch(buildUrl("/api/plan"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(await parseError(response, `Planning request failed with status ${response.status}`));
  }

  return response.json();
}

export async function requestExportMp4(payload) {
  const response = await fetch(buildUrl("/api/export-mp4"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(await parseError(response, `MP4 export failed with status ${response.status}`));
  }

  const blob = await response.blob();
  const disposition = response.headers.get("Content-Disposition") ?? "";
  const match = disposition.match(/filename=\"?([^\";]+)\"?/i);
  const filename = match?.[1] || "rrt_animation.mp4";
  return { blob, filename };
}
