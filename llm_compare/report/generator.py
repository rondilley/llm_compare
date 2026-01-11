"""PDF report generator for evaluation sessions."""

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, ListFlowable, ListItem, KeepTogether, Flowable
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

from ..utils.logging import get_logger
from ..session.manager import Session
from ..prompting.repetition import get_repetition_info

logger = get_logger(__name__)


class BoxedParagraphs(Flowable):
    """A flowable that wraps paragraphs in a bordered box that can split across pages."""

    def __init__(self, paragraphs: List[Paragraph], width: float,
                 bg_color=None, border_color=None, padding: float = 10):
        Flowable.__init__(self)
        self.paragraphs = paragraphs
        self.width = width
        self.bg_color = bg_color or colors.white
        self.border_color = border_color or colors.HexColor('#dee2e6')
        self.padding = padding
        self._calculated_height = None

    def wrap(self, availWidth, availHeight):
        """Calculate the space needed."""
        self.width = min(self.width, availWidth)
        content_width = self.width - 2 * self.padding

        total_height = 0
        for para in self.paragraphs:
            w, h = para.wrap(content_width, availHeight)
            total_height += h

        self._calculated_height = total_height + 2 * self.padding
        return self.width, self._calculated_height

    def split(self, availWidth, availHeight):
        """Split the flowable if it doesn't fit."""
        if self._calculated_height is None:
            self.wrap(availWidth, availHeight)

        if self._calculated_height <= availHeight:
            return [self]

        # Just return paragraphs as individual elements with styling
        # This allows them to flow naturally across pages
        result = []
        for i, para in enumerate(self.paragraphs):
            result.append(para)
        return result

    def draw(self):
        """Draw the boxed content."""
        canvas = self.canv

        # Draw background
        canvas.setFillColor(self.bg_color)
        canvas.setStrokeColor(self.border_color)
        canvas.rect(0, 0, self.width, self._calculated_height, fill=1, stroke=1)

        # Draw paragraphs
        y = self._calculated_height - self.padding
        content_width = self.width - 2 * self.padding

        for para in self.paragraphs:
            w, h = para.wrap(content_width, 10000)
            y -= h
            para.drawOn(canvas, self.padding, y)


class ReportGenerator:
    """Generates PDF reports from evaluation sessions."""

    def __init__(self):
        """Initialize the report generator."""
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Set up custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.HexColor('#2c3e50'),
        ))

        self.styles.add(ParagraphStyle(
            name='SubHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=8,
            textColor=colors.HexColor('#34495e'),
        ))

        self.styles.add(ParagraphStyle(
            name='ResponseText',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=14,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        ))

        self.styles.add(ParagraphStyle(
            name='SmallText',
            parent=self.styles['Normal'],
            fontSize=9,
            leading=12,
        ))

        self.styles.add(ParagraphStyle(
            name='CenterBold',
            parent=self.styles['Normal'],
            fontSize=12,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
        ))

        # Markdown-specific styles
        self.styles.add(ParagraphStyle(
            name='MarkdownH1',
            parent=self.styles['Heading1'],
            fontSize=14,
            spaceAfter=8,
            spaceBefore=12,
            textColor=colors.HexColor('#2c3e50'),
        ))

        self.styles.add(ParagraphStyle(
            name='MarkdownH2',
            parent=self.styles['Heading2'],
            fontSize=12,
            spaceAfter=6,
            spaceBefore=10,
            textColor=colors.HexColor('#34495e'),
        ))

        self.styles.add(ParagraphStyle(
            name='MarkdownH3',
            parent=self.styles['Heading3'],
            fontSize=11,
            spaceAfter=4,
            spaceBefore=8,
            textColor=colors.HexColor('#495057'),
        ))

        self.styles.add(ParagraphStyle(
            name='CodeBlock',
            parent=self.styles['Normal'],
            fontName='Courier',
            fontSize=9,
            leading=11,
            backColor=colors.HexColor('#f4f4f4'),
            borderColor=colors.HexColor('#ddd'),
            borderWidth=1,
            borderPadding=6,
            spaceAfter=8,
            spaceBefore=8,
        ))

        self.styles.add(ParagraphStyle(
            name='BulletItem',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=14,
            leftIndent=20,
            bulletIndent=10,
            spaceAfter=3,
        ))

        self.styles.add(ParagraphStyle(
            name='NumberedItem',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=14,
            leftIndent=20,
            bulletIndent=10,
            spaceAfter=3,
        ))

    def generate(self, session: Session, output_path: Path) -> Path:
        """
        Generate a PDF report for a session.

        Args:
            session: Completed evaluation session
            output_path: Path for output PDF

        Returns:
            Path to generated PDF
        """
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch,
        )

        story = []

        # Cover page
        story.extend(self._build_cover(session))
        story.append(PageBreak())

        # Section 1: Original Prompt
        story.extend(self._build_prompt_section(session))
        story.append(Spacer(1, 0.3*inch))

        # Section 1.5: Repetition Analysis (if enabled)
        if session.repetition_analysis:
            story.extend(self._build_repetition_section(session))
            story.append(Spacer(1, 0.3*inch))

        # Section 2: Responses
        story.extend(self._build_responses_section(session))
        story.append(PageBreak())

        # Section 3: Pointwise Evaluation
        if session.pointwise_results:
            story.extend(self._build_pointwise_section(session))
            story.append(Spacer(1, 0.3*inch))

        # Section 4: Pairwise Comparison
        if session.pairwise_results:
            story.extend(self._build_pairwise_section(session))
            story.append(PageBreak())

        # Section 5: Adversarial Debate
        if session.adversarial_results:
            story.extend(self._build_adversarial_section(session))
            story.append(PageBreak())

        # Section 6: Collaborative Consensus
        if session.consensus_results:
            story.extend(self._build_consensus_section(session))
            story.append(Spacer(1, 0.3*inch))

        # Section 7: Final Rankings
        if session.final_rankings:
            story.extend(self._build_rankings_section(session))

        # Build PDF
        doc.build(story)
        logger.info(f"Report generated: {output_path}")

        return output_path

    def _build_cover(self, session: Session) -> List:
        """Build cover page."""
        elements = []

        elements.append(Spacer(1, 2*inch))

        # Title
        title = Paragraph(
            "LLM Comparison Report",
            ParagraphStyle(
                name='Title',
                fontSize=28,
                alignment=TA_CENTER,
                fontName='Helvetica-Bold',
                textColor=colors.HexColor('#2c3e50'),
            )
        )
        elements.append(title)
        elements.append(Spacer(1, 0.5*inch))

        # Session info
        elements.append(Paragraph(
            f"Session ID: {session.session_id}",
            self.styles['CenterBold']
        ))
        elements.append(Spacer(1, 0.2*inch))

        elements.append(Paragraph(
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            ParagraphStyle(
                name='DateStyle',
                fontSize=11,
                alignment=TA_CENTER,
            )
        ))
        elements.append(Spacer(1, 0.5*inch))

        # Providers used
        providers = list(session.responses.keys())
        elements.append(Paragraph(
            f"Providers Evaluated: {', '.join(providers)}",
            ParagraphStyle(
                name='ProvidersStyle',
                fontSize=11,
                alignment=TA_CENTER,
            )
        ))

        # Duration
        if session.metadata.total_duration_ms:
            duration_s = session.metadata.total_duration_ms / 1000
            elements.append(Spacer(1, 0.2*inch))
            elements.append(Paragraph(
                f"Total Duration: {duration_s:.1f} seconds",
                ParagraphStyle(
                    name='DurationStyle',
                    fontSize=11,
                    alignment=TA_CENTER,
                )
            ))

        return elements

    def _build_prompt_section(self, session: Session) -> List:
        """Build prompt section."""
        elements = []

        elements.append(Paragraph("1. Original Prompt", self.styles['SectionHeader']))

        # Prompt text - escape and format
        prompt_text = self._escape_xml(session.prompt)
        prompt_text = prompt_text.replace('\n', '<br/>')

        # Use a styled paragraph with background via custom style
        prompt_style = ParagraphStyle(
            name='PromptText',
            parent=self.styles['ResponseText'],
            backColor=colors.HexColor('#f8f9fa'),
            borderColor=colors.HexColor('#dee2e6'),
            borderWidth=1,
            borderPadding=12,
        )
        elements.append(Paragraph(prompt_text, prompt_style))

        return elements

    def _build_repetition_section(self, session: Session) -> List:
        """Build prompt repetition analysis section."""
        elements = []

        analysis = session.repetition_analysis
        if not analysis:
            return elements

        elements.append(Paragraph("1.5. Prompt Repetition", self.styles['SectionHeader']))

        # Mode used
        mode_info = get_repetition_info(analysis.mode_used)
        elements.append(Paragraph(
            f"<b>Mode Used:</b> {mode_info['name']}",
            self.styles['Normal']
        ))
        elements.append(Paragraph(
            f"{mode_info['description']}",
            self.styles['SmallText']
        ))
        elements.append(Spacer(1, 0.1*inch))

        # Recommendation
        if analysis.recommended_mode:
            rec_info = get_repetition_info(analysis.recommended_mode)
            elements.append(Paragraph(
                f"<b>Recommended Mode:</b> {rec_info['name']}",
                self.styles['Normal']
            ))
            elements.append(Paragraph(
                f"Reason: {analysis.recommendation_reason}",
                self.styles['SmallText']
            ))
            elements.append(Spacer(1, 0.1*inch))

        # Comparison table if available
        if analysis.compare_enabled and analysis.provider_comparisons:
            elements.append(Paragraph(
                "Baseline vs. Repeated Prompt Comparison:",
                self.styles['SubHeader']
            ))
            elements.append(Spacer(1, 0.1*inch))

            # Build comparison table
            data = [['Provider', 'Baseline\nLatency', 'Repeated\nLatency', 'Delta', 'Baseline\nLength', 'Repeated\nLength', 'Preferred']]

            for provider, comp in analysis.provider_comparisons.items():
                delta_pct = comp.get('latency_delta_pct', 0)
                delta_str = f"{comp.get('latency_delta_ms', 0):+d}ms\n({delta_pct:+.1f}%)"

                data.append([
                    provider.upper(),
                    f"{comp.get('baseline_latency_ms', 0)}ms",
                    f"{comp.get('repeated_latency_ms', 0)}ms",
                    delta_str,
                    str(comp.get('baseline_length', 0)),
                    str(comp.get('repeated_length', 0)),
                    comp.get('preferred_mode', 'N/A').capitalize(),
                ])

            table = Table(data, colWidths=[1*inch, 0.9*inch, 0.9*inch, 0.85*inch, 0.85*inch, 0.85*inch, 0.85*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 8),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
                ('PADDING', (0, 0), (-1, -1), 4),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            elements.append(table)

            elements.append(Spacer(1, 0.1*inch))
            elements.append(Paragraph(
                "<i>Note: Based on the paper \"Prompt Repetition Improves Non-Reasoning LLMs\" "
                "(Leviathan et al., 2025), repeating prompts allows each token to attend to every "
                "other token, improving performance without increasing output length.</i>",
                self.styles['SmallText']
            ))

        return elements

    def _build_responses_section(self, session: Session) -> List:
        """Build responses section."""
        elements = []

        elements.append(Paragraph("2. Responses", self.styles['SectionHeader']))

        for provider, response in session.responses.items():
            elements.append(Paragraph(
                f"2.{list(session.responses.keys()).index(provider)+1}. {provider.upper()}",
                self.styles['SubHeader']
            ))

            # Metadata
            meta_text = (
                f"Model: {response.model_id} | "
                f"Latency: {response.latency_ms}ms | "
                f"Tokens: {response.input_tokens + response.output_tokens}"
            )
            elements.append(Paragraph(meta_text, self.styles['SmallText']))
            elements.append(Spacer(1, 0.1*inch))

            # Convert markdown to PDF elements
            md_elements = self._markdown_to_flowables(response.text)
            elements.extend(md_elements)

            elements.append(Spacer(1, 0.3*inch))

        return elements

    def _escape_xml(self, text: str) -> str:
        """Escape special XML characters for ReportLab."""
        if not text:
            return ""
        # Replace ampersand first to avoid double-escaping
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        return text

    def _markdown_to_flowables(self, text: str) -> List:
        """Convert markdown text to ReportLab flowables."""
        elements = []

        if not text:
            return elements

        # Split into lines for processing
        lines = text.split('\n')
        current_paragraph = []
        in_code_block = False
        code_block_lines = []
        code_language = ""

        i = 0
        while i < len(lines):
            line = lines[i]

            # Check for code block start/end
            if line.strip().startswith('```'):
                if in_code_block:
                    # End of code block
                    if code_block_lines:
                        code_text = self._escape_xml('\n'.join(code_block_lines))
                        code_text = code_text.replace('\n', '<br/>')
                        elements.append(Paragraph(code_text, self.styles['CodeBlock']))
                    code_block_lines = []
                    in_code_block = False
                else:
                    # Start of code block - flush current paragraph first
                    if current_paragraph:
                        para_text = self._process_inline_markdown(' '.join(current_paragraph))
                        if para_text.strip():
                            elements.append(Paragraph(para_text, self.styles['ResponseText']))
                        current_paragraph = []
                    in_code_block = True
                    code_language = line.strip()[3:].strip()
                i += 1
                continue

            if in_code_block:
                code_block_lines.append(line)
                i += 1
                continue

            # Check for headers
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                # Flush current paragraph
                if current_paragraph:
                    para_text = self._process_inline_markdown(' '.join(current_paragraph))
                    if para_text.strip():
                        elements.append(Paragraph(para_text, self.styles['ResponseText']))
                    current_paragraph = []

                level = len(header_match.group(1))
                header_text = self._escape_xml(header_match.group(2))
                if level == 1:
                    elements.append(Paragraph(f"<b>{header_text}</b>", self.styles['MarkdownH1']))
                elif level == 2:
                    elements.append(Paragraph(f"<b>{header_text}</b>", self.styles['MarkdownH2']))
                else:
                    elements.append(Paragraph(f"<b>{header_text}</b>", self.styles['MarkdownH3']))
                i += 1
                continue

            # Check for bullet points
            bullet_match = re.match(r'^[\s]*[-*+]\s+(.+)$', line)
            if bullet_match:
                # Flush current paragraph
                if current_paragraph:
                    para_text = self._process_inline_markdown(' '.join(current_paragraph))
                    if para_text.strip():
                        elements.append(Paragraph(para_text, self.styles['ResponseText']))
                    current_paragraph = []

                bullet_text = self._process_inline_markdown(bullet_match.group(1))
                elements.append(Paragraph(f"• {bullet_text}", self.styles['BulletItem']))
                i += 1
                continue

            # Check for numbered lists
            numbered_match = re.match(r'^[\s]*(\d+)[.)]\s+(.+)$', line)
            if numbered_match:
                # Flush current paragraph
                if current_paragraph:
                    para_text = self._process_inline_markdown(' '.join(current_paragraph))
                    if para_text.strip():
                        elements.append(Paragraph(para_text, self.styles['ResponseText']))
                    current_paragraph = []

                num = numbered_match.group(1)
                item_text = self._process_inline_markdown(numbered_match.group(2))
                elements.append(Paragraph(f"{num}. {item_text}", self.styles['NumberedItem']))
                i += 1
                continue

            # Empty line - end of paragraph
            if not line.strip():
                if current_paragraph:
                    para_text = self._process_inline_markdown(' '.join(current_paragraph))
                    if para_text.strip():
                        elements.append(Paragraph(para_text, self.styles['ResponseText']))
                    current_paragraph = []
                i += 1
                continue

            # Regular text - add to current paragraph
            current_paragraph.append(line.strip())
            i += 1

        # Flush remaining paragraph
        if current_paragraph:
            para_text = self._process_inline_markdown(' '.join(current_paragraph))
            if para_text.strip():
                elements.append(Paragraph(para_text, self.styles['ResponseText']))

        # Handle unclosed code block
        if in_code_block and code_block_lines:
            code_text = self._escape_xml('\n'.join(code_block_lines))
            code_text = code_text.replace('\n', '<br/>')
            elements.append(Paragraph(code_text, self.styles['CodeBlock']))

        return elements

    def _process_inline_markdown(self, text: str) -> str:
        """Process inline markdown formatting (bold, italic, code, links)."""
        # Escape XML first
        text = self._escape_xml(text)

        # Bold: **text** or __text__
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)

        # Italic: *text* or _text_ (but not inside words)
        text = re.sub(r'(?<!\w)\*([^*]+?)\*(?!\w)', r'<i>\1</i>', text)
        text = re.sub(r'(?<!\w)_([^_]+?)_(?!\w)', r'<i>\1</i>', text)

        # Inline code: `code`
        text = re.sub(r'`([^`]+?)`', r'<font face="Courier" size="9">\1</font>', text)

        # Links: [text](url) - just show the text
        text = re.sub(r'\[([^\]]+?)\]\([^)]+?\)', r'<u>\1</u>', text)

        return text

    def _build_pointwise_section(self, session: Session) -> List:
        """Build pointwise evaluation section."""
        elements = []

        elements.append(Paragraph("3. Pointwise Evaluation", self.styles['SectionHeader']))
        elements.append(Paragraph(
            "Each response was scored independently against evaluation rubrics by all other providers.",
            self.styles['Normal']
        ))
        elements.append(Spacer(1, 0.2*inch))

        # Build score table
        results = session.pointwise_results
        providers = list(results.evaluations.keys())

        # Header row
        header = ['Provider', 'Accuracy', 'Completeness', 'Clarity', 'Relevance', 'Reasoning', 'Overall']
        data = [header]

        for provider in providers:
            eval_result = results.evaluations[provider]
            row = [provider.upper()]
            for rubric in ['accuracy', 'completeness', 'clarity', 'relevance', 'reasoning']:
                score = eval_result.aggregated_scores.get(rubric, 0)
                row.append(f"{score:.1f}")
            row.append(f"{eval_result.overall_score:.2f}")
            data.append(row)

        table = Table(data, colWidths=[1.2*inch] + [0.85*inch]*6)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(table)

        return elements

    def _build_pairwise_section(self, session: Session) -> List:
        """Build pairwise comparison section."""
        elements = []

        elements.append(Paragraph("4. Pairwise Comparison", self.styles['SectionHeader']))
        elements.append(Paragraph(
            "Head-to-head comparisons between all pairs of responses.",
            self.styles['Normal']
        ))
        elements.append(Spacer(1, 0.2*inch))

        results = session.pairwise_results
        providers = list(results.win_rates.keys())

        # Win rate summary
        elements.append(Paragraph("Win Rates:", self.styles['SubHeader']))

        data = [['Provider', 'Win Rate']]
        sorted_rates = sorted(results.win_rates.items(), key=lambda x: x[1], reverse=True)
        for provider, rate in sorted_rates:
            data.append([provider.upper(), f"{rate:.1%}"])

        table = Table(data, colWidths=[2*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(table)

        return elements

    def _build_adversarial_section(self, session: Session) -> List:
        """Build adversarial debate section."""
        elements = []

        elements.append(Paragraph("5. Adversarial Debate", self.styles['SectionHeader']))
        elements.append(Paragraph(
            "Each response was subjected to structured debate with advocates and challengers.",
            self.styles['Normal']
        ))
        elements.append(Spacer(1, 0.2*inch))

        results = session.adversarial_results

        for debate in results.debates:
            elements.append(Paragraph(
                f"Debate: {debate.response_id.upper()} Response",
                self.styles['SubHeader']
            ))

            # Roles
            role_text = (
                f"Advocate: {debate.advocate} | "
                f"Challenger: {debate.challenger} | "
                f"Judge: {debate.judge}"
            )
            elements.append(Paragraph(role_text, self.styles['SmallText']))
            elements.append(Spacer(1, 0.1*inch))

            # Verdict summary
            if debate.verdict:
                verdict = debate.verdict
                elements.append(Paragraph(
                    f"Final Score: {verdict.score:.1f}/10",
                    ParagraphStyle(
                        name='VerdictScore',
                        fontSize=12,
                        fontName='Helvetica-Bold',
                        textColor=colors.HexColor('#27ae60') if verdict.score >= 7 else colors.HexColor('#e74c3c'),
                    )
                ))

                if verdict.validated_strengths:
                    elements.append(Paragraph("Validated Strengths:", self.styles['SmallText']))
                    for strength in verdict.validated_strengths[:3]:
                        elements.append(Paragraph(f"  - {strength}", self.styles['SmallText']))

                if verdict.confirmed_weaknesses:
                    elements.append(Paragraph("Confirmed Weaknesses:", self.styles['SmallText']))
                    for weakness in verdict.confirmed_weaknesses[:3]:
                        elements.append(Paragraph(f"  - {weakness}", self.styles['SmallText']))

            elements.append(Spacer(1, 0.2*inch))

        return elements

    def _build_consensus_section(self, session: Session) -> List:
        """Build collaborative consensus section."""
        elements = []

        elements.append(Paragraph("6. Collaborative Consensus", self.styles['SectionHeader']))
        elements.append(Paragraph(
            "Multi-model discussion to identify shared insights and build consensus.",
            self.styles['Normal']
        ))
        elements.append(Spacer(1, 0.2*inch))

        results = session.consensus_results

        # Shared strengths
        if results.shared_strengths:
            elements.append(Paragraph("Shared Strengths:", self.styles['SubHeader']))
            for insight in results.shared_strengths[:5]:
                text = f"- {insight.description} (Confidence: {insight.confidence:.0%})"
                elements.append(Paragraph(text, self.styles['SmallText']))
            elements.append(Spacer(1, 0.1*inch))

        # Shared weaknesses
        if results.shared_weaknesses:
            elements.append(Paragraph("Shared Weaknesses:", self.styles['SubHeader']))
            for insight in results.shared_weaknesses[:5]:
                text = f"- {insight.description} (Confidence: {insight.confidence:.0%})"
                elements.append(Paragraph(text, self.styles['SmallText']))
            elements.append(Spacer(1, 0.1*inch))

        # Synthesis
        if results.synthesis:
            elements.append(Paragraph("Synthesis:", self.styles['SubHeader']))
            elements.append(Paragraph(results.synthesis, self.styles['ResponseText']))

        return elements

    def _build_rankings_section(self, session: Session) -> List:
        """Build final rankings section."""
        elements = []

        elements.append(Paragraph("7. Final Rankings", self.styles['SectionHeader']))
        elements.append(Paragraph(
            "Final rankings computed using Bradley-Terry model with weighted aggregation of all evaluation phases.",
            self.styles['Normal']
        ))
        elements.append(Spacer(1, 0.2*inch))

        rankings = session.final_rankings

        # Rankings table
        data = [['Rank', 'Provider', 'Score', '95% CI', 'Pointwise', 'Pairwise', 'Debate', 'Consensus']]

        for ranked in rankings.rankings:
            breakdown = ranked.score_breakdown
            row = [
                str(ranked.rank),
                ranked.provider.upper(),
                f"{ranked.score:.2f}",
                f"[{ranked.confidence_interval[0]:.2f}, {ranked.confidence_interval[1]:.2f}]",
                f"{breakdown.get('pointwise', 0):.2f}",
                f"{breakdown.get('pairwise_winrate', 0):.2f}",
                f"{breakdown.get('debate', 0):.2f}",
                f"{breakdown.get('consensus', 0):.2f}",
            ]
            data.append(row)

        table = Table(data, colWidths=[0.5*inch, 1*inch, 0.7*inch, 1.3*inch, 0.85*inch, 0.85*inch, 0.7*inch, 0.85*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
            ('PADDING', (0, 0), (-1, -1), 5),
            # Highlight winner
            ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#d4edda')),
        ]))
        elements.append(table)

        elements.append(Spacer(1, 0.3*inch))

        # Winner announcement
        winner = rankings.rankings[0]
        elements.append(Paragraph(
            f"Winner: {winner.provider.upper()} with score {winner.score:.2f}",
            ParagraphStyle(
                name='Winner',
                fontSize=14,
                alignment=TA_CENTER,
                fontName='Helvetica-Bold',
                textColor=colors.HexColor('#27ae60'),
            )
        ))

        return elements
