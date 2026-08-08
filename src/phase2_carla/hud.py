import time
import pygame


def draw_hud(screen, font, wp, speed_kmh, risk_level,
             alert_level, face_detected, ear, mar, head_tilt_deg):
    drowsy_critical = alert_level == "ALERT_CRITICAL"
    drowsy_warn = alert_level == "ALERT_WARN"

    drowsiness_label = (
        "DROWSY!" if drowsy_critical else
        "WARNING" if drowsy_warn else
        "NO FACE" if not face_detected else "OK"
    )
    drowsiness_color = (
        (255, 0, 0) if drowsy_critical else
        (255, 165, 0) if drowsy_warn else
        (200, 200, 200) if not face_detected else (0, 255, 0)
    )

    hud = [
        (f"Speed: {speed_kmh:.1f} km/h",                            (255, 255, 255)),
        (f"Risk: {risk_level}",                                      (255, 255, 255)),
        (f"Junction: {'YES' if wp.is_junction else 'NO'}",           (255, 255, 255)),
        (f"Lane: {wp.lane_id} ({str(wp.lane_type).split('.')[-1]})", (255, 255, 255)),
        ("── Driver Monitor ──",                                     (180, 180, 180)),
        (f"Driver: {drowsiness_label}",                              drowsiness_color),
        (f"EAR:{ear:.2f}  MAR:{mar:.2f}",                           drowsiness_color),
        (f"Head tilt: {head_tilt_deg:.1f}°",                        drowsiness_color),
    ]

    for i, (text, color) in enumerate(hud):
        screen.blit(font.render(text, True, color), (10, 10 + i * 22))


def draw_mode_indicator(screen, font, auto_mode):
    label = f"MODE: {'AUTO' if auto_mode else 'MANUAL'}"
    screen.blit(font.render(label, True, (255, 255, 255)), (10, 10))


def draw_alerts(screen, alert_font, alert_level, risk_level, WIDTH, HEIGHT):
    flash = int(time.time() * 4) % 2 == 0

    if alert_level == "ALERT_CRITICAL" and flash:
        surf = alert_font.render("DRIVER DROWSY - AUTO BRAKE!", True, (255, 50, 50))
        screen.blit(surf, (WIDTH // 2 - surf.get_width() // 2, HEIGHT // 4 + 60))
    elif alert_level == "ALERT_WARN" and flash:
        surf = alert_font.render("DROWSINESS WARNING", True, (255, 165, 0))
        screen.blit(surf, (WIDTH // 2 - surf.get_width() // 2, HEIGHT // 4 + 60))

    if risk_level == "EMERGENCY" and flash:
        surf = alert_font.render("EMERGENCY BRAKE!", True, (255, 0, 0))
        screen.blit(surf, (WIDTH // 2 - surf.get_width() // 2, HEIGHT // 4))
