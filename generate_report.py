import os
import csv
import glob
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, HRFlowable, Image, PageBreak
)
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics import renderPDF
import tempfile

# ─── COLORS ───────────────────────────────────────────────
AU_BLUE      = colors.HexColor("#2c3e6b")
AU_RED       = colors.HexColor("#8b1a1a")
AU_LIGHT     = colors.HexColor("#dce8f5")
AU_GREY      = colors.HexColor("#f7f9fc")
AU_BORDER    = colors.HexColor("#dde3ec")
GREEN        = colors.HexColor("#27ae60")
ORANGE       = colors.HexColor("#e67e22")
RED_ALERT    = colors.HexColor("#e74c3c")
BLUE_CHART   = colors.HexColor("#2980b9")
TEXT_DARK    = colors.HexColor("#2d2d2d")
TEXT_MUTED   = colors.HexColor("#888888")

EMOTION_HEX = {
    'Happy':     '#27ae60',
    'Surprised': '#e67e22',
    'Neutral':   '#7f8c8d',
    'Fearful':   '#8e44ad',
    'Disgusted': '#16a085',
    'Sad':       '#2980b9',
    'Angry':     '#e74c3c',
}

ENGAGEMENT_SCORES = {
    'Happy':1.0,'Surprised':0.9,'Neutral':0.5,
    'Fearful':0.3,'Disgusted':0.2,'Sad':0.1,'Angry':0.1,
}

# ─── READ CSV ─────────────────────────────────────────────
def read_session_csv(csv_path):
    rows = []
    with open(csv_path, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

# ─── COMPUTE STATS ────────────────────────────────────────
def compute_stats(rows):
    if not rows:
        return {}

    emotions      = [r.get('Emotion','Neutral') for r in rows]
    scores        = [float(r.get('Engagement_Score', ENGAGEMENT_SCORES.get(r.get('Emotion','Neutral'),0.5))) for r in rows]
    timestamps    = [r.get('Timestamp','') for r in rows]

    emotion_counts = {}
    for e in emotions:
        emotion_counts[e] = emotion_counts.get(e,0) + 1

    total         = len(rows)
    avg_score     = sum(scores)/total if total else 0
    focused       = sum(1 for s in scores if s >= 0.5)
    dominant      = max(emotion_counts, key=emotion_counts.get)
    low_periods   = sum(1 for s in scores if s < 0.35)

    if avg_score >= 0.75:   eng_label = "Highly Engaged"
    elif avg_score >= 0.5:  eng_label = "Moderate"
    elif avg_score >= 0.3:  eng_label = "Low Focus"
    else:                   eng_label = "Disengaged"

    return {
        'total':         total,
        'avg_score':     avg_score,
        'eng_label':     eng_label,
        'focused':       focused,
        'focused_pct':   round(focused/total*100) if total else 0,
        'dominant':      dominant,
        'low_periods':   low_periods,
        'emotion_counts':emotion_counts,
        'scores':        scores,
        'timestamps':    timestamps,
        'emotions':      emotions,
    }

# ─── CHART 1: Engagement Timeline ─────────────────────────
def make_timeline_chart(scores, timestamps):
    fig, ax = plt.subplots(figsize=(7, 2.2))
    x = list(range(len(scores)))
    ax.fill_between(x, scores, alpha=0.25, color='#2980b9')
    ax.plot(x, scores, color='#2980b9', linewidth=1.8)
    ax.axhline(0.5, color='#e67e22', linewidth=0.8, linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, max(len(scores)-1, 1))
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0%','25%','50%','75%','100%'], fontsize=7)
    ax.set_xlabel('Sample Number', fontsize=7)
    ax.set_ylabel('Engagement', fontsize=7)
    ax.set_title('Engagement Score Over Session', fontsize=9, fontweight='bold')
    ax.tick_params(labelsize=7)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    plt.savefig(tmp.name, dpi=150, bbox_inches='tight')
    plt.close()
    return tmp.name

# ─── CHART 2: Emotion Pie Chart ───────────────────────────
def make_emotion_pie(emotion_counts):
    labels = list(emotion_counts.keys())
    values = list(emotion_counts.values())
    clrs   = [EMOTION_HEX.get(e,'#999999') for e in labels]

    fig, ax = plt.subplots(figsize=(4, 3))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=clrs,
        autopct='%1.0f%%', startangle=140,
        pctdistance=0.8,
        textprops={'fontsize': 7}
    )
    for at in autotexts:
        at.set_fontsize(7)
    ax.set_title('Emotion Distribution', fontsize=9, fontweight='bold')
    plt.tight_layout()
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    plt.savefig(tmp.name, dpi=150, bbox_inches='tight')
    plt.close()
    return tmp.name

# ─── CHART 3: Engagement Bar ──────────────────────────────
def make_engagement_bar(emotion_counts):
    emotions = list(emotion_counts.keys())
    counts   = list(emotion_counts.values())
    clrs     = [EMOTION_HEX.get(e,'#999999') for e in emotions]

    fig, ax = plt.subplots(figsize=(4, 2.5))
    bars = ax.barh(emotions, counts, color=clrs, height=0.55)
    ax.set_xlabel('Count', fontsize=7)
    ax.set_title('Emotion Frequency', fontsize=9, fontweight='bold')
    ax.tick_params(labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_width()+0.1, bar.get_y()+bar.get_height()/2,
                str(cnt), va='center', fontsize=7)
    plt.tight_layout()
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    plt.savefig(tmp.name, dpi=150, bbox_inches='tight')
    plt.close()
    return tmp.name

# ─── HEADER / FOOTER ──────────────────────────────────────
def make_header_footer(canvas, doc):
    canvas.saveState()
    W, H = A4

    # Header bar
    canvas.setFillColor(AU_BLUE)
    canvas.rect(0, H-18*mm, W, 18*mm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 11)
    canvas.drawString(15*mm, H-11*mm, "ALLIANCE UNIVERSITY")
    canvas.setFont("Helvetica", 8)
    canvas.drawString(15*mm, H-16*mm, "School of Engineering & Technology — EduSense AI Report")
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(W-15*mm, H-11*mm, "NAAC Grade A+")
    canvas.drawRightString(W-15*mm, H-16*mm,
        datetime.now().strftime("Generated: %d %b %Y, %H:%M"))

    # Red accent line
    canvas.setFillColor(AU_RED)
    canvas.rect(0, H-19.5*mm, W, 1.5*mm, fill=1, stroke=0)

    # Footer
    canvas.setFillColor(AU_BLUE)
    canvas.rect(0, 0, W, 10*mm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica", 7)
    canvas.drawString(15*mm, 3.5*mm,
        "Alliance University, Bengaluru — 562106 | EduSense AI Classroom Monitor | Confidential")
    canvas.drawRightString(W-15*mm, 3.5*mm, f"Page {doc.page}")

    canvas.restoreState()

# ─── STYLES ───────────────────────────────────────────────
def get_styles():
    styles = getSampleStyleSheet()
    return {
        'title': ParagraphStyle('title',
            fontSize=18, textColor=AU_BLUE,
            fontName='Helvetica-Bold', spaceAfter=4,
            alignment=TA_LEFT),
        'subtitle': ParagraphStyle('subtitle',
            fontSize=10, textColor=TEXT_MUTED,
            fontName='Helvetica', spaceAfter=12),
        'section': ParagraphStyle('section',
            fontSize=11, textColor=AU_BLUE,
            fontName='Helvetica-Bold', spaceBefore=10, spaceAfter=6,
            borderPad=4),
        'normal': ParagraphStyle('normal',
            fontSize=9, textColor=TEXT_DARK,
            fontName='Helvetica', spaceAfter=4, leading=14),
        'small': ParagraphStyle('small',
            fontSize=8, textColor=TEXT_MUTED,
            fontName='Helvetica', spaceAfter=2),
        'rec': ParagraphStyle('rec',
            fontSize=9, textColor=TEXT_DARK,
            fontName='Helvetica', spaceAfter=6,
            leftIndent=10, leading=14),
    }

# ─── GENERATE RECOMMENDATIONS ─────────────────────────────
def get_recommendations(stats):
    recs = []
    avg  = stats['avg_score']
    dom  = stats['dominant']

    if avg >= 0.75:
        recs.append("Class engagement was excellent. Maintain current teaching pace and methodology.")
        recs.append("Consider introducing advanced topics or interactive activities to sustain momentum.")
    elif avg >= 0.5:
        recs.append("Moderate engagement observed. Incorporate more interactive activities or polls.")
        recs.append("Consider brief 5-minute breaks every 30 minutes to maintain focus levels.")
    elif avg >= 0.3:
        recs.append("Low engagement detected. Review lecture pacing and content difficulty.")
        recs.append("Use multimedia content, case studies or group activities to re-engage students.")
        recs.append("Consider checking with students individually for understanding gaps.")
    else:
        recs.append("Critical: Class showed very low engagement throughout. Immediate intervention needed.")
        recs.append("Review if content is too advanced or unclear. Consider a revision session.")
        recs.append("Speak with individual students to identify specific challenges.")

    if dom in ['Sad','Angry','Fearful']:
        recs.append(f"Dominant emotion was '{dom}' — check for external stressors (exams, workload).")
    if stats['low_periods'] > stats['total'] * 0.4:
        recs.append("Over 40% of session had low engagement — consider shorter, focused sessions.")

    return recs

# ─── BUILD PDF ────────────────────────────────────────────
def generate_report(csv_path, output_path, staff_name="Faculty",
                    staff_role="Faculty", dept="CSE — AI & Data Science",
                    course="Cognitive AI (E1CSC 25608)", room="Lab 204"):

    rows  = read_session_csv(csv_path)
    stats = compute_stats(rows)

    if not stats:
        print("[ERROR] No data found in CSV.")
        return

    S = get_styles()

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        topMargin=22*mm, bottomMargin=14*mm,
        leftMargin=15*mm, rightMargin=15*mm,
        title="EduSense Session Report",
        author="Alliance University EduSense AI"
    )

    story = []

    # ── Cover info ────────────────────────────────────────
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph("EduSense AI — Session Report", S['title']))
    story.append(Paragraph(
        f"Classroom Engagement Analysis &nbsp;|&nbsp; {datetime.now().strftime('%d %B %Y')}",
        S['subtitle']))
    story.append(HRFlowable(width="100%", thickness=1.5,
                             color=AU_RED, spaceAfter=8))

    # ── Session info table ────────────────────────────────
    session_data = [
        ["Staff Name", staff_name,      "Department",  dept],
        ["Role",       staff_role,       "Course",      course],
        ["Date",       datetime.now().strftime("%d %b %Y"),
                                         "Room / Lab",  room],
        ["Total Samples", str(stats['total']),
                                         "Data Source", os.path.basename(csv_path)],
    ]

    info_table = Table(session_data, colWidths=[30*mm, 55*mm, 32*mm, 58*mm])
    info_table.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (0,-1), AU_LIGHT),
        ('BACKGROUND',  (2,0), (2,-1), AU_LIGHT),
        ('TEXTCOLOR',   (0,0), (0,-1), AU_BLUE),
        ('TEXTCOLOR',   (2,0), (2,-1), AU_BLUE),
        ('FONTNAME',    (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME',    (2,0), (2,-1), 'Helvetica-Bold'),
        ('FONTSIZE',    (0,0), (-1,-1), 8),
        ('GRID',        (0,0), (-1,-1), 0.5, AU_BORDER),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.white, AU_GREY]),
        ('PADDING',     (0,0), (-1,-1), 5),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 6*mm))

    # ── Summary stat boxes ────────────────────────────────
    story.append(Paragraph("Session Summary", S['section']))

    avg_pct    = round(stats['avg_score']*100)
    eng_color  = GREEN if avg_pct>=70 else (ORANGE if avg_pct>=40 else RED_ALERT)

    summary_data = [[
        Paragraph(f"<b><font size=20>{avg_pct}%</font></b><br/>"
                  f"<font size=8 color='#888888'>Avg Engagement</font>", S['normal']),
        Paragraph(f"<b><font size=20>{stats['focused_pct']}%</font></b><br/>"
                  f"<font size=8 color='#888888'>Students Focused</font>", S['normal']),
        Paragraph(f"<b><font size=14>{stats['eng_label']}</font></b><br/>"
                  f"<font size=8 color='#888888'>Overall Status</font>", S['normal']),
        Paragraph(f"<b><font size=20>{stats['total']}</font></b><br/>"
                  f"<font size=8 color='#888888'>Total Samples</font>", S['normal']),
        Paragraph(f"<b><font size=14>{stats['dominant']}</font></b><br/>"
                  f"<font size=8 color='#888888'>Dominant Emotion</font>", S['normal']),
    ]]

    summary_table = Table(summary_data,
                          colWidths=[35*mm, 35*mm, 42*mm, 30*mm, 33*mm])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,-1), AU_GREY),
        ('BOX',         (0,0), (-1,-1), 1, AU_BORDER),
        ('GRID',        (0,0), (-1,-1), 0.5, AU_BORDER),
        ('ALIGN',       (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING',     (0,0), (-1,-1), 10),
        ('TOPPADDING',  (0,0), (-1,-1), 10),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 6*mm))

    # ── Charts ────────────────────────────────────────────
    story.append(Paragraph("Engagement Analysis", S['section']))

    timeline_img = make_timeline_chart(stats['scores'], stats['timestamps'])
    pie_img      = make_emotion_pie(stats['emotion_counts'])
    bar_img      = make_engagement_bar(stats['emotion_counts'])

    # Timeline full width
    story.append(Image(timeline_img, width=175*mm, height=50*mm))
    story.append(Spacer(1, 4*mm))

    # Pie + Bar side by side
    chart_row = [[
        Image(pie_img, width=85*mm, height=60*mm),
        Image(bar_img, width=85*mm, height=60*mm),
    ]]
    chart_table = Table(chart_row, colWidths=[88*mm, 88*mm])
    chart_table.setStyle(TableStyle([
        ('ALIGN',  (0,0),(-1,-1),'CENTER'),
        ('VALIGN', (0,0),(-1,-1),'MIDDLE'),
    ]))
    story.append(chart_table)
    story.append(Spacer(1, 6*mm))

    # ── Emotion breakdown table ───────────────────────────
    story.append(Paragraph("Emotion Breakdown", S['section']))

    emo_header = [
        Paragraph("<b>Emotion</b>", S['small']),
        Paragraph("<b>Count</b>",   S['small']),
        Paragraph("<b>Percentage</b>", S['small']),
        Paragraph("<b>Eng. Score</b>", S['small']),
        Paragraph("<b>Classification</b>", S['small']),
    ]
    emo_rows   = [emo_header]
    total_s    = stats['total']

    for em, cnt in sorted(stats['emotion_counts'].items(),
                          key=lambda x:-x[1]):
        pct   = round(cnt/total_s*100)
        score = ENGAGEMENT_SCORES.get(em, 0.5)
        cls   = ("Engaged" if score>=0.7
                 else "Moderate" if score>=0.5
                 else "Low" if score>=0.3
                 else "Disengaged")
        emo_rows.append([
            Paragraph(em, S['normal']),
            Paragraph(str(cnt), S['normal']),
            Paragraph(f"{pct}%", S['normal']),
            Paragraph(f"{round(score*100)}%", S['normal']),
            Paragraph(cls, S['normal']),
        ])

    emo_table = Table(emo_rows,
                      colWidths=[35*mm, 22*mm, 35*mm, 35*mm, 48*mm])
    emo_table.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0), AU_LIGHT),
        ('TEXTCOLOR',     (0,0), (-1,0), AU_BLUE),
        ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0,0), (-1,-1), 8),
        ('GRID',          (0,0), (-1,-1), 0.5, AU_BORDER),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [colors.white, AU_GREY]),
        ('ALIGN',         (1,0), (-1,-1), 'CENTER'),
        ('PADDING',       (0,0), (-1,-1), 5),
    ]))
    story.append(emo_table)
    story.append(Spacer(1, 6*mm))

    # ── Recommendations ───────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=AU_BORDER))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph("Recommendations", S['section']))

    recs = get_recommendations(stats)
    for i, rec in enumerate(recs, 1):
        story.append(Paragraph(f"{i}.  {rec}", S['rec']))

    story.append(Spacer(1, 4*mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=AU_BORDER))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        "This report was auto-generated by EduSense AI using MobileNetV2 emotion recognition. "
        "Results are indicative and should be interpreted alongside direct classroom observation.",
        S['small']))

    # ── Build ─────────────────────────────────────────────
    doc.build(story, onFirstPage=make_header_footer,
              onLaterPages=make_header_footer)

    # Cleanup temp chart images
    for f in [timeline_img, pie_img, bar_img]:
        try: os.unlink(f)
        except: pass

    print(f"\n[OK] Report saved: {output_path}")

# ─── CLI ──────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EduSense — Generate PDF session report from CSV")
    parser.add_argument("--csv",   default=None,
        help="Path to session CSV (default: latest in ./sessions/)")
    parser.add_argument("--out",   default=None,
        help="Output PDF path (default: ./sessions/report_<timestamp>.pdf)")
    parser.add_argument("--name",  default="Dr. R. Sharma",
        help="Staff name")
    parser.add_argument("--role",  default="Faculty",
        help="Staff role")
    parser.add_argument("--dept",  default="CSE - AI & Data Science",
        help="Department")
    parser.add_argument("--course",default="Cognitive AI (E1CSC 25608)",
        help="Course name")
    parser.add_argument("--room",  default="Lab 204",
        help="Room or lab name")
    args = parser.parse_args()

    # Auto find latest CSV
    if args.csv is None:
        csvs = sorted(glob.glob("./sessions/*.csv"))
        if not csvs:
            print("[ERROR] No CSV files found in ./sessions/")
            exit(1)
        args.csv = csvs[-1]
        print(f"[INFO] Using latest session: {args.csv}")

    if args.out is None:
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = f"./sessions/EduSense_Report_{ts}.pdf"

    generate_report(
        csv_path    = args.csv,
        output_path = args.out,
        staff_name  = args.name,
        staff_role  = args.role,
        dept        = args.dept,
        course      = args.course,
        room        = args.room,
    )
