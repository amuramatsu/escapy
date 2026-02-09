#  EscaPy is a software allowing to convert EPSON ESC/P, ESC/P2
#  printer control language files into PDF files.
#  Copyright (C) 2024-2025  Ysard
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Tests for ESC/P 24-J84C (Japanese) support"""
# Standard imports
from functools import partial
from pathlib import Path

# Custom imports
import pytest

# Local imports
from escapy.grammar import init_parser
from escapy.parser import (
    ESCParser as _ESCParser,
    J84_HANKAKU_WIDTH,
    J84_ZENKAKU_WIDTH,
    PrintScripting,
    decode_j84_text,
    group_j84_runs,
)
from .misc import default_printer_profile

# Do not touch the filesystem fonts: the tests below never build a PDF.
TEST_FONTS = {
    0: {
        "fixed": lambda *_: None,
        "proportional": lambda *_: None,
    }
}
ESCParser = partial(
    _ESCParser,
    printer_profile=default_printer_profile,
    available_fonts=TEST_FONTS,
)

FS = b"\x1c"
KANJI_A = b"$\""  # JIS X 0208 0x24 0x22 -> "あ"
FS_AND = FS + b"&"
FS_DOT = FS + b"."
# Default intercharacter spaces from the ESC/P 24-J84C manual.
DEFAULT_ZENKAKU_SPACE = 3 / 180
DEFAULT_HANKAKU_SPACE = 2 / 180


def tree_names(tree):
    """Return the sorted list of rule names contained in a parse tree."""
    return [subtree.data for subtree in tree.iter_subtrees()]


def build_block(*parts):
    """Build a ``FS & ... FS .`` block from the given parts."""
    return FS_AND + b"".join(parts) + FS_DOT


def test_kanji_block_after_text():
    """A kanji block can follow text on the same line."""
    tree = init_parser(b"coucou" + build_block(KANJI_A))

    assert "kanji_print" in tree_names(tree)
    assert "binary_blob" in tree_names(tree)


def test_kanji_block_at_start():
    """A kanji block at the beginning of the stream is printed."""
    tree = init_parser(build_block(KANJI_A) + b"coucou")

    assert "kanji_print" in tree_names(tree)
    assert "binary_blob" in tree_names(tree)


def test_kanji_does_not_leak_into_text():
    """KANJI tokens must not be produced outside a kanji block."""
    tree = init_parser(b"coucou")

    assert "kanji_print" not in tree_names(tree)


def test_short_text_is_not_kanji():
    """Two printable bytes outside a block stay plain text."""
    for payload in (b"ab", b"ab\r\n", b'$"'):
        tree = init_parser(payload)

        assert "kanji_print" not in tree_names(tree)
        assert "binary_blob" in tree_names(tree)


def test_j84_command_outside_block():
    """FS commands are accepted outside kanji blocks too."""
    escapy = ESCParser(FS + b"\x0f" + build_block(KANJI_A), pdf=False)

    assert escapy.hankaku_mode is True
    assert escapy.cursor_x == pytest.approx(
        escapy.left_margin + J84_HANKAKU_WIDTH + DEFAULT_HANKAKU_SPACE
    )


def test_empty_kanji_block():
    """An empty kanji block is accepted and does nothing."""
    tree = init_parser(build_block())

    assert "kanji_block" in tree_names(tree)
    assert "kanji_print" not in tree_names(tree)


def test_multiline_kanji_block():
    """A kanji block can span several lines."""
    tree = init_parser(build_block(KANJI_A + b"\r\n" + KANJI_A))

    assert tree_names(tree).count("kanji_print") == 2
    assert "carriage_return" in tree_names(tree)
    assert "line_feed" in tree_names(tree)


def test_kanji_print_moves_cursor():
    """A zenkaku character advances the cursor by a 24-dot cell and 3 dots."""
    escapy = ESCParser(build_block(KANJI_A), pdf=False)

    assert escapy.cursor_x == pytest.approx(
        escapy.left_margin + J84_ZENKAKU_WIDTH + DEFAULT_ZENKAKU_SPACE
    )


def test_halfwidth_kana_print_moves_cursor():
    """Two half-width kana characters use two 12-dot cells and 2-dot spaces."""
    escapy = ESCParser(build_block(b"\xb1\xb2"), pdf=False)

    assert escapy.cursor_x == pytest.approx(
        escapy.left_margin
        + 2 * (J84_HANKAKU_WIDTH + DEFAULT_HANKAKU_SPACE)
    )


def test_double_width_printing():
    """FS SO doubles the width of the characters and their spacing."""
    escapy = ESCParser(build_block(FS + b"\x0e" + KANJI_A), pdf=False)

    assert escapy.cursor_x == pytest.approx(
        escapy.left_margin
        + 2 * (J84_ZENKAKU_WIDTH + DEFAULT_ZENKAKU_SPACE)
    )


def test_quad_size_printing():
    """FS W 1 doubles both the width and the height."""
    escapy = ESCParser(build_block(FS + b"W1" + KANJI_A), pdf=False)

    assert escapy.cursor_x == pytest.approx(
        escapy.left_margin
        + 2 * (J84_ZENKAKU_WIDTH + DEFAULT_ZENKAKU_SPACE)
    )


def test_zenkaku_intercharacter_space():
    """FS S sets the left and right spaces of zenkaku characters."""
    escapy = ESCParser(build_block(FS + b"S\x02\x00" + KANJI_A), pdf=False)

    assert escapy.jp_zenkaku_space_L == 2
    assert escapy.jp_zenkaku_space_R == 0
    assert escapy.cursor_x == pytest.approx(
        escapy.left_margin + J84_ZENKAKU_WIDTH + 2 / 180
    )


def test_hankaku_intercharacter_space():
    """FS T sets the left and right spaces of half-width characters."""
    escapy = ESCParser(build_block(FS + b"T\x03\x04" + b"\xb1\xb2"), pdf=False)

    assert escapy.jp_hankaku_space_L == 3
    assert escapy.jp_hankaku_space_R == 4
    assert escapy.cursor_x == pytest.approx(
        escapy.left_margin + 2 * (J84_HANKAKU_WIDTH + 7 / 180)
    )


def test_intercharacter_space_doubles_with_double_width():
    """The manual specifies doubled spacing during horizontal expansion."""
    escapy = ESCParser(
        build_block(FS + b"S\x02\x03" + FS + b"\x0e" + KANJI_A), pdf=False
    )

    assert escapy.cursor_x == pytest.approx(
        escapy.left_margin + 2 * (J84_ZENKAKU_WIDTH + 5 / 180)
    )


def test_halfwidth_kanji_printing_and_cancel():
    """FS SI shrinks zenkaku glyphs; FS DC2 restores normal width."""
    escapy = ESCParser(
        build_block(FS + b"\x0f" + KANJI_A + FS + b"\x12" + KANJI_A), pdf=False
    )

    assert escapy.hankaku_mode is False
    assert escapy.cursor_x == pytest.approx(
        escapy.left_margin
        + J84_HANKAKU_WIDTH
        + DEFAULT_HANKAKU_SPACE
        + J84_ZENKAKU_WIDTH
        + DEFAULT_ZENKAKU_SPACE
    )


def test_halfwidth_and_script_cancel_each_other():
    """FS SI cancels scripts; FS DC2 cancels both half-width and scripts."""
    escapy = ESCParser(build_block(FS + b"r1" + FS + b"\x0f" + KANJI_A), pdf=False)

    assert escapy.hankaku_mode is True
    assert escapy.jp_scripting is None

    escapy = ESCParser(build_block(FS + b"r1" + FS + b"\x12" + KANJI_A), pdf=False)

    assert escapy.hankaku_mode is False
    assert escapy.jp_scripting is None


def test_double_width_printing_cancel():
    """FS DC4 cancels only the double-width mode."""
    escapy = ESCParser(
        build_block(FS + b"\x0e" + KANJI_A + FS + b"\x14" + KANJI_A), pdf=False
    )

    assert escapy.double_width is False
    assert escapy.cursor_x == pytest.approx(
        escapy.left_margin
        + 2 * (J84_ZENKAKU_WIDTH + DEFAULT_ZENKAKU_SPACE)
        + J84_ZENKAKU_WIDTH
        + DEFAULT_ZENKAKU_SPACE
    )


@pytest.mark.parametrize("line_end", [b"\n", b"\r\n"])
def test_double_width_printing_is_cancelled_at_line_end(line_end):
    """FS SO is canceled at the end of the line."""
    escapy = ESCParser(
        build_block(FS + b"\x0e" + KANJI_A + line_end + KANJI_A), pdf=False
    )

    assert escapy.double_width is False
    assert escapy.cursor_x == pytest.approx(
        escapy.left_margin + J84_ZENKAKU_WIDTH + DEFAULT_ZENKAKU_SPACE
    )


def test_quad_size_printing_survives_line_end():
    """FS W is a multiline mode and is not canceled at the end of the line."""
    escapy = ESCParser(
        build_block(FS + b"W1" + KANJI_A + b"\r\n" + KANJI_A), pdf=False
    )

    assert escapy.double_width is True
    assert escapy.jp_double_height is True
    assert escapy.cursor_x == pytest.approx(
        escapy.left_margin
        + 2 * (J84_ZENKAKU_WIDTH + DEFAULT_ZENKAKU_SPACE)
    )


def test_quad_size_printing_cancel():
    """FS W 0 cancels the 2x2 size mode."""
    escapy = ESCParser(
        build_block(FS + b"W1" + KANJI_A + FS + b"W0" + KANJI_A), pdf=False
    )

    assert escapy.double_width is False
    assert escapy.jp_double_height is False
    assert escapy.cursor_x == pytest.approx(
        escapy.left_margin
        + 2 * (J84_ZENKAKU_WIDTH + DEFAULT_ZENKAKU_SPACE)
        + J84_ZENKAKU_WIDTH
        + DEFAULT_ZENKAKU_SPACE
    )


def test_superscript_printing():
    """FS r 0 selects superscript and uses half-size character cells."""
    escapy = ESCParser(build_block(FS + b"r0" + KANJI_A), pdf=False)

    assert escapy.jp_scripting is PrintScripting.SUP
    assert escapy.cursor_x == pytest.approx(
        escapy.left_margin + J84_HANKAKU_WIDTH + DEFAULT_HANKAKU_SPACE
    )


@pytest.mark.parametrize(
    "arg, expected",
    [
        (b"0", 0),
        (b"1", 1),
        (b"2", 2),
        (b"\x00", 0),
        (b"\x01", 1),
        (b"\x02", 2),
    ],
)
def test_underline_values(arg, expected):
    """FS - accepts the 1-dot and 2-dot underline thicknesses."""
    escapy = ESCParser(build_block(FS + b"-" + arg + KANJI_A), pdf=False)

    assert escapy.jp_underline == expected


def test_hankaku_space_adjustment():
    """FS U adds the half-width adjustment; FS V removes it."""
    escapy = ESCParser(build_block(FS + b"U" + b"\xb1"), pdf=False)

    assert escapy.jp_hankaku_zenkaku_adjust is True
    assert escapy.cursor_x == pytest.approx(
        escapy.left_margin + J84_HANKAKU_WIDTH + DEFAULT_HANKAKU_SPACE + 0.5 / 180
    )

    escapy = ESCParser(build_block(FS + b"U" + FS + b"V" + b"\xb1"), pdf=False)

    assert escapy.jp_hankaku_zenkaku_adjust is False
    assert escapy.cursor_x == pytest.approx(
        escapy.left_margin + J84_HANKAKU_WIDTH + DEFAULT_HANKAKU_SPACE
    )


def test_tategaki_printing_commands():
    """FS J and FS K switch vertical and horizontal writing modes."""
    escapy = ESCParser(build_block(FS + b"J" + FS + b"K"), pdf=False)
    assert escapy.jp_tategaki is False

    escapy = ESCParser(
        build_block(FS + b"J" + FS + b"D\x01\x02\x03\x04"), pdf=False
    )
    assert escapy.jp_tategaki is True
    assert escapy.jp_hankaku_tate_kumi == [1, 2, 3, 4]


def test_typeface_selection_can_be_reset():
    """FS k 0 restores the Mincho typeface."""
    escapy = ESCParser(build_block(FS + b"k1" + FS + b"k0" + KANJI_A), pdf=False)

    assert escapy.jp_typeface == "mincho"


@pytest.mark.parametrize(
    "value, tategaki, hankaku, double_width, double_height, scripting, underline",
    [
        (0x00, False, False, False, False, None, 0),
        (0x01, True, False, False, False, None, 0),
        (0x02, False, True, False, False, None, 0),
        (0x04, False, False, True, False, None, 0),
        (0x08, False, False, False, True, None, 0),
        (0x10, False, False, False, False, PrintScripting.SUP, 0),
        (0x30, False, False, False, False, PrintScripting.SUB, 0),
        (0x12, False, True, False, False, PrintScripting.SUP, 0),
        (0x80, False, False, False, False, None, 1),
        (0x40, False, False, False, False, None, 0),
    ],
)
def test_master_select_bits(
    value, tategaki, hankaku, double_width, double_height, scripting, underline
):
    """FS ! applies each bit of the master-select byte."""
    escapy = ESCParser(build_block(FS + b"!" + bytes((value,))), pdf=False)

    assert escapy.jp_tategaki is tategaki
    assert escapy.hankaku_mode is hankaku
    assert escapy.double_width is double_width
    assert escapy.jp_double_height is double_height
    assert escapy.jp_scripting is scripting
    assert escapy.jp_underline == underline


def test_master_select_script_priority():
    """The scripting bit cancels half-width kanji."""
    escapy = ESCParser(
        build_block(FS + b"\x0f" + FS + b"!\x10" + KANJI_A), pdf=False
    )

    assert escapy.jp_scripting is PrintScripting.SUP


def test_master_select_preserves_underline_thickness():
    """FS ! enables the underline without losing the FS - thickness."""
    escapy = ESCParser(
        build_block(FS + b"-2" + FS + b"!\x80" + KANJI_A), pdf=False
    )

    assert escapy.jp_underline == 2

    escapy = ESCParser(build_block(FS + b"!\x80" + KANJI_A), pdf=False)

    assert escapy.jp_underline == 1

    escapy = ESCParser(
        build_block(FS + b"-2" + FS + b"!\x00" + KANJI_A), pdf=False
    )

    assert escapy.jp_underline == 0


def test_master_select_double_height_rendering():
    """FS ! bit 3 doubles the height but keeps the character width."""
    escapy = ESCParser(build_block(FS + b"!\x08" + KANJI_A), pdf=False)

    assert escapy.double_width is False
    assert escapy.jp_double_height is True
    assert escapy.cursor_x == pytest.approx(
        escapy.left_margin + J84_ZENKAKU_WIDTH + DEFAULT_ZENKAKU_SPACE
    )


def test_j84_commands_are_idempotent():
    """FS commands can be repeated without turning methods into booleans."""
    code = build_block(FS + b"\x0f", FS + b"\x0f", KANJI_A)
    escapy = ESCParser(code, pdf=False)

    assert escapy.hankaku_mode is True


def test_all_j84_commands_are_idempotent():
    """Every FS command can be called more than once."""
    bitmap = bytes(range(72))
    commands = [
        FS + b"\x0f",
        FS + b"\x12",
        FS + b"\x0e",
        FS + b"\x14",
        FS + b"r0",
        FS + b"x0",
        FS + b"x1",
        FS + b"W0",
        FS + b"J",
        FS + b"K",
        FS + b"D\x00\x00\x00\x00",
        FS + b"-0",
        FS + b"!\x00",
        FS + b"k0",
        FS + b"S\x00\x00",
        FS + b"T\x00\x00",
        FS + b"U",
        FS + b"V",
        FS + b"2w!" + bitmap,
    ]
    escapy = ESCParser(build_block(*(commands * 2)), pdf=False)

    assert escapy.gaiji[(0x77, 0x21)] == bitmap


def test_switch_script_printing_jp():
    """FS r selects superscript (0) or subscript (1)."""
    escapy = ESCParser(build_block(FS + b"r1" + KANJI_A), pdf=False)

    assert escapy.jp_scripting is PrintScripting.SUB


def test_select_letter_quality_or_draft_jp():
    """FS x is parsed both inside and outside a kanji block."""
    for code in (FS + b"x0", FS + b"x1", build_block(FS + b"x1")):
        assert "select_letter_quality_or_draft_jp" in tree_names(
            init_parser(code)
        )


def test_select_typeface_jp():
    """FS k 1 selects the Gothic typeface."""
    escapy = ESCParser(build_block(FS + b"k1" + KANJI_A), pdf=False)

    assert escapy.jp_typeface == "gothic"


def test_kanji_font_selection():
    """Half-width kana use their own font when one is configured."""
    escapy = ESCParser(build_block(KANJI_A), pdf=False)
    escapy.kanji_fontname = "Mincho"
    escapy.kanji_fontname_gothic = "Gothic"

    assert escapy._kanji_fontname() == "Mincho"
    assert escapy._kanji_fontname(0.5) == "Mincho"

    escapy.jp_typeface = "gothic"
    assert escapy._kanji_fontname() == "Gothic"
    assert escapy._kanji_fontname(0.5) == "Gothic"

    escapy.hankaku_fontname = "Kana"
    assert escapy._kanji_fontname() == "Gothic"
    assert escapy._kanji_fontname(0.5) == "Kana"


def test_configured_ttf_fonts_are_registered():
    """Configured TTF files override the built-in CID fonts."""
    import reportlab
    from reportlab.pdfbase import pdfmetrics

    vera = Path(reportlab.__file__).parent / "fonts" / "Vera.ttf"
    escapy = ESCParser(
        b"a",
        japanese_fonts={
            "mincho": str(vera),
            "gothic": str(vera),
            "hankaku": str(vera),
        },
        pdf=False,
    )
    escapy._register_japanese_fonts()

    expected = _ESCParser._ttf_fontname(vera)
    assert escapy.kanji_fontname == expected
    assert escapy.kanji_fontname_gothic == expected
    assert escapy.hankaku_fontname == expected
    assert expected in pdfmetrics.getRegisteredFontNames()


def test_mincho_ttf_is_used_for_gothic_by_default():
    """With only mincho configured, FS k 1 uses the same TTF font."""
    import reportlab

    vera = Path(reportlab.__file__).parent / "fonts" / "Vera.ttf"
    escapy = ESCParser(
        b"a",
        japanese_fonts={"mincho": str(vera)},
        pdf=False,
    )
    escapy._register_japanese_fonts()

    expected = _ESCParser._ttf_fontname(vera)
    assert escapy.kanji_fontname == expected
    assert escapy.kanji_fontname_gothic == expected


def test_invalid_ttf_font_falls_back_to_cid():
    """An invalid configured font keeps the built-in CID fallback."""
    escapy = ESCParser(
        b"a",
        japanese_fonts={"mincho": "/does/not/exist.ttf"},
        pdf=False,
    )
    escapy._register_japanese_fonts()

    assert escapy.kanji_fontname == "HeiseiMin-W3"
    assert escapy.hankaku_fontname is None


def test_unsupported_fs_command_is_ignored():
    """An unknown FS command is skipped instead of aborting the parsing."""
    expected = pytest.approx(J84_ZENKAKU_WIDTH + DEFAULT_ZENKAKU_SPACE)
    for code in (
        FS + b"0" + build_block(KANJI_A),
        build_block(FS + b"0" + KANJI_A),
    ):
        escapy = ESCParser(code, pdf=False)

        assert escapy.cursor_x - escapy.left_margin == expected


def test_master_select_jp():
    """FS ! treats 0x1c as its argument, not as a command prefix."""
    escapy = ESCParser(build_block(FS + b"!\x1c" + KANJI_A), pdf=False)

    assert escapy.double_width is True
    assert escapy.jp_double_height is True
    assert escapy.jp_scripting is None


def test_define_gaiji():
    """FS 2 stores the 72-byte bitmap of a user-defined character."""
    bitmap = bytes(range(72))
    code = build_block(FS + b"2w!" + bitmap, KANJI_A)
    escapy = ESCParser(code, pdf=False)

    assert escapy.gaiji[(0x77, 0x21)] == bitmap


@pytest.mark.parametrize(
    "raw_text, expected_text, expected_widths",
    [
        (KANJI_A, "あ", [1.0]),
        (b"\xb1\xb2", "ｱｲ", [0.5, 0.5]),
        (b"\x20", " ", [0.5]),
        (b"\x77\x21", "〓", [1.0]),
    ],
)
def test_decode_j84_text(raw_text, expected_text, expected_widths):
    """JIS/half-width kana bytes are decoded with the right cell widths."""
    text, widths = decode_j84_text(raw_text)

    assert text == expected_text
    assert widths == expected_widths


def test_group_j84_runs():
    """Consecutive characters with the same width are grouped."""
    text, widths = decode_j84_text(KANJI_A + b"\xb1\xb2")

    assert group_j84_runs(text, widths) == [("あ", 1.0), ("ｱｲ", 0.5)]
