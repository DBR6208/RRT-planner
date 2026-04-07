from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "timestamp_utc" in payload


def test_cspace_endpoint_returns_grid_and_collision_flags() -> None:
    payload = {
        "start": [0.0, 0.0],
        "goal": [1.57079632679, 0.0],
        "obstacle": {"x": 0.8, "y": 0.0, "size": 0.2},
        "collision_buffer": 0.0,
        "cspace_resolution": 60,
    }

    response = client.post("/api/cspace", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert data["cspace_resolution"] == 60
    assert len(data["cspace"]) == 60
    assert len(data["cspace"][0]) == 60

    assert data["start_in_collision"] is True
    assert data["goal_in_collision"] is False


def test_plan_endpoint_supports_wrap_toggle() -> None:
    base_payload = {
        "start": [3.04159, 0.0],
        "goal": [-3.04159, 0.0],
        "obstacle": {"x": 3.0, "y": 3.0, "size": 0.2},
        "expand_dist": 0.2,
        "goal_rate": 0.15,
        "max_iter": 4000,
        "edge_check_steps": 20,
        "collision_buffer": 0.0,
        "include_cspace": False,
        "seed": 7,
    }

    wrapped = client.post("/api/plan", json={**base_payload, "wrap_angles": True})
    non_wrapped = client.post("/api/plan", json={**base_payload, "wrap_angles": False})

    assert wrapped.status_code == 200
    assert non_wrapped.status_code == 200

    wrapped_payload = wrapped.json()
    non_wrapped_payload = non_wrapped.json()

    assert wrapped_payload["wrap_angles"] is True
    assert non_wrapped_payload["wrap_angles"] is False

    assert isinstance(wrapped_payload["tree"], list)
    assert isinstance(non_wrapped_payload["tree"], list)
    assert len(wrapped_payload["tree"]) == len(wrapped_payload["parents"])
    assert len(non_wrapped_payload["tree"]) == len(non_wrapped_payload["parents"])

    assert wrapped_payload["success"] is True
    assert non_wrapped_payload["success"] is True


def test_cspace_endpoint_supports_joint_limits_and_link_lengths() -> None:
    payload = {
        "start": [2.5, 1.0],
        "goal": [-2.5, -1.0],
        "obstacle": {"x": 0.8, "y": 0.0, "size": 0.2},
        "joint_limits": {
            "theta1_min": -1.2,
            "theta1_max": 1.2,
            "theta2_min": -0.9,
            "theta2_max": 0.9,
        },
        "link_lengths": {"l1": 0.8, "l2": 0.6},
        "collision_buffer": 0.0,
        "cspace_resolution": 48,
    }

    response = client.post("/api/cspace", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["cspace_resolution"] == 48
    assert len(data["cspace"]) == 48
    assert data["joint_limits"]["theta1_min"] == -1.2
    assert data["joint_limits"]["theta1_max"] == 1.2
    assert data["joint_limits"]["theta2_min"] == -0.9
    assert data["joint_limits"]["theta2_max"] == 0.9
    assert data["link_lengths"]["l1"] == 0.8
    assert data["link_lengths"]["l2"] == 0.6

    # start/goal are clipped into the configured joint limits in cspace checks
    assert data["start"] == [1.2, 0.9]
    assert data["goal"] == [-1.2, -0.9]


def test_plan_endpoint_disables_wrap_for_non_periodic_joint_limits() -> None:
    payload = {
        "start": [0.2, 0.0],
        "goal": [2.8, 0.0],
        "obstacle": {"x": 3.0, "y": 3.0, "size": 0.2},
        "joint_limits": {
            "theta1_min": 0.0,
            "theta1_max": 3.14159265359,
            "theta2_min": -3.14159265359,
            "theta2_max": 3.14159265359,
        },
        "link_lengths": {"l1": 1.0, "l2": 1.0},
        "expand_dist": 0.2,
        "goal_rate": 0.2,
        "max_iter": 5000,
        "edge_check_steps": 20,
        "collision_buffer": 0.0,
        "wrap_angles": True,
        "include_cspace": False,
        "seed": 7,
    }

    response = client.post("/api/plan", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert data["wrap_angles"] is False
    theta1_values = [q[0] for q in data["path"]]
    assert min(theta1_values) >= -1e-9
    assert max(theta1_values) <= 3.14159265359 + 1e-9


def test_export_mp4_endpoint_streams_video(monkeypatch) -> None:
    from app import main as main_module

    def fake_export_bytes(_request) -> bytes:
        return b"\x00\x00fake-mp4"

    monkeypatch.setattr(main_module, "generate_mp4_bytes", fake_export_bytes)

    payload = {
        "start": [0.0, 0.0],
        "goal": [1.0, 0.0],
        "obstacle": {"x": 0.9, "y": 0.2, "size": 0.2},
        "joint_limits": {
            "theta1_min": -3.14159265359,
            "theta1_max": 3.14159265359,
            "theta2_min": -3.14159265359,
            "theta2_max": 3.14159265359,
        },
        "link_lengths": {"l1": 1.0, "l2": 1.0},
        "expand_dist": 0.03,
        "goal_rate": 0.1,
        "max_iter": 5000,
        "edge_check_steps": 20,
        "collision_buffer": 0.05,
        "wrap_angles": True,
        "include_cspace": True,
        "cspace_resolution": 120,
        "seed": 42,
        "fps": 24,
        "dpi": 110,
        "bitrate": 2200,
        "samples_per_edge": 2,
    }

    response = client.post("/api/export-mp4", json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("video/mp4")
    assert "attachment" in response.headers.get("content-disposition", "")
    assert response.content == b"\x00\x00fake-mp4"
