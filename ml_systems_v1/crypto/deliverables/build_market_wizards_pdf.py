#!/usr/bin/env python3
"""Build 'The Market Wizards Series -- Stories & Distilled Lessons' as a single PDF (fpdf2).

Expanded edition: fans out the STORIES behind the lessons across Jack Schwager's six Market Wizards books.
Run: python deliverables/build_market_wizards_pdf.py  ->  deliverables/Market_Wizards_Distilled.pdf
"""
from __future__ import annotations
from fpdf import FPDF

INK = (20, 20, 20); ACCENT = (10, 70, 140); GREY = (90, 90, 90)


def _ascii(s: str) -> str:
    return (s.replace("—", "--").replace("–", "-").replace("’", "'").replace("‘", "'")
             .replace("“", '"').replace("”", '"').replace("…", "...").replace("→", "->")
             .encode("latin-1", "replace").decode("latin-1"))


class PDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8); self.set_text_color(*GREY)
        self.cell(0, 8, "The Market Wizards Series -- Stories & Lessons", align="L")
        self.cell(0, 8, f"p.{self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(200, 200, 200); self.line(self.l_margin, 18, self.w - self.r_margin, 18); self.ln(4)

    def h1(self, t):
        self.set_x(self.l_margin); self.set_font("Helvetica", "B", 16); self.set_text_color(*ACCENT)
        self.multi_cell(0, 8, _ascii(t), new_x="LMARGIN", new_y="NEXT"); self.ln(1); self.set_text_color(*INK)

    def h2(self, t):
        self.ln(1.5); self.set_x(self.l_margin); self.set_font("Helvetica", "B", 12.5); self.set_text_color(*ACCENT)
        self.multi_cell(0, 6.5, _ascii(t), new_x="LMARGIN", new_y="NEXT"); self.set_text_color(*INK)

    def name(self, t):
        self.ln(0.8); self.set_x(self.l_margin); self.set_font("Helvetica", "B", 10.5); self.set_text_color(*INK)
        self.multi_cell(0, 5.2, _ascii(t), new_x="LMARGIN", new_y="NEXT")

    def para(self, t, size=10.3, style=""):
        self.set_x(self.l_margin); self.set_font("Helvetica", style, size)
        self.multi_cell(0, 5.3, _ascii(t), new_x="LMARGIN", new_y="NEXT"); self.ln(1.0)

    def quote(self, t, who):
        self.set_x(self.l_margin); self.set_font("Helvetica", "I", 10.3); self.set_text_color(*GREY)
        self.multi_cell(0, 5.2, _ascii(f'"{t}"  -- {who}'), new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*INK); self.ln(0.6)


BOOKS = [
    ("1. Market Wizards (1989)",
     "The book that started it all. Schwager hunted down the traders whispered about on the futures floors and in "
     "the back offices of Wall Street and simply asked them how they did it. The answers were wildly different in "
     "method and almost identical in spirit.",
     [("Michael Marcus", "Marcus began with almost nothing and a weakness for hot tips -- he borrowed against "
       "everything, followed a 'sure thing' in soybeans, and was wiped out, twice, early on. What saved him was "
       "Ed Seykota, who hired him and taught him to ride winners and respect risk. Over roughly a decade at "
       "Commodities Corp he compounded about $30,000 into some $80 million. His hardest-won lesson: size up only "
       "when conviction and risk both agree, and never let a tip override your own analysis. 'The best trades are "
       "the ones in which you have all three things going for you: fundamentals, technicals, and market tone.'"),
      ("Bruce Kovner", "A Harvard man who had driven a cab before his first trade -- a leveraged soybean spread that "
       "turned $3,000 into $40,000, then collapsed to $23,000 in days. That whipsaw taught him everything: "
       "'Undertrade, undertrade, undertrade.' He sizes so a normal loss is trivial, and places stops where the "
       "market would prove his idea WRONG -- not at the point of maximum dollar pain. He grew Caxton into one of "
       "the largest macro funds in the world on that discipline."),
      ("Paul Tudor Jones", "Brash, intense, and famous for calling and trading the October 1987 crash (captured in "
       "the cult documentary 'Trader'), turning that single quarter into a legend. He is obsessed with defense: "
       "targets roughly 5-to-1 reward-to-risk, cuts size the moment he is losing, and re-marks every position to "
       "the current price as if he entered today. 'Don't be a hero. Don't have an ego. Always question yourself and "
       "your ability.' And: 'I'm always thinking about losing money as opposed to making money.'"),
      ("Ed Seykota", "The quiet pioneer who, in the 1970s, built some of the first computerized trend-following "
       "systems. One managed account reportedly grew several thousand-fold over ~16 years. He trades from home, "
       "follows his rules, and philosophizes: 'Everybody gets what they want out of the markets.' His rule set, "
       "delivered deadpan: 'The elements of good trading are cutting losses, cutting losses, and cutting losses.'"),
      ("Marty Schwartz", "'Pit Bull.' For nearly a decade he was a losing securities analyst, broke and miserable. "
       "The turnaround was not a new indicator -- it was money management and the decision to stop letting his ego "
       "trade. He then won the U.S. Investing Championship repeatedly, with audited triple-digit returns. 'The most "
       "important change in my trading career was when I learned to divorce my ego from the trade -- to admit when "
       "I was wrong and get out.'"),
      ("Richard Dennis (and the Turtles)", "Dennis turned a few hundred borrowed dollars into a fortune estimated "
       "near $200 million. To settle an argument with his partner William Eckhardt over whether great trading is "
       "born or made, he recruited a class of novices -- the 'Turtles' -- taught them a mechanical trend-following "
       "system over two weeks, and staked them. Many became millionaire traders. The experiment proved his point: "
       "discipline and a tested edge, followed faithfully, can be taught."),
      ("Larry Hite", "Built Mint into a giant on one idea -- risk control as religion. 'There are two basic rules "
       "about winning in trading as well as in life: 1) If you don't bet, you can't win. 2) If you lose all your "
       "chips, you can't bet.' He never risked more than ~1% on a trade, and treated the system, not his opinion, "
       "as the boss.")]),
    ("2. The New Market Wizards (1992)",
     "A second expedition, now reaching into FX, options, and short-term trading. The counterintuitive truths come "
     "into sharper focus -- chiefly that the comfortable instinct is usually the losing one.",
     [("Bill Lipschutz", "Salomon Brothers' legendary FX trader, reportedly earning the firm hundreds of millions a "
       "year. He turned a $12,000 inheritance into a fortune trading stocks while still a student -- then lost "
       "nearly all of it, a loss he calls the best lesson he ever bought. He trades enormous size with surgical "
       "risk and patience: 'If most traders would learn to sit on their hands 50 percent of the time, they would "
       "make a lot more money.'"),
      ("William Eckhardt", "Dennis's mathematician partner and the brains behind the Turtle rules. His theme is the "
       "psychology of self-sabotage: 'What feels good is often the wrong thing to do.' Taking small profits feels "
       "responsible and quietly caps your winners; holding losers feels patient and quietly destroys you. The edge "
       "lives in doing the uncomfortable, correct thing repeatedly."),
      ("Linda Bradford Raschke", "A rare woman on these pages and a master of short-term technical trading, who "
       "recovered from a serious injury to build a decades-long career. Her message is unglamorous: 'I don't think "
       "I'm one of those geniuses... I just work very, very hard.' Edge is preparation and discipline, applied to "
       "a few well-understood patterns."),
      ("Monroe Trout & Randy McKay", "Trout posted years of high returns with astonishingly low drawdowns by "
       "treating risk statistically and never reaching for a trade. McKay's lesson is the mirror image of cutting "
       "losses: ride the rare huge winners hard, and shrink your size to nothing whenever you are trading badly -- "
       "let the market tell you when you are 'on.'")]),
    ("3. Stock Market Wizards (2001)",
     "Equity specialists at the peak and collapse of the dot-com era. The dominant lesson of the book is sobering: "
     "edges decay. Methods that minted money in one regime quietly stop working, and only the adaptable survive -- "
     "while the risk rules never change.",
     [("The recurring arc", "Several wizards describe a strategy that worked beautifully... until it didn't. The "
       "survivors noticed early, cut back, and rebuilt the method -- but they never loosened position sizing or "
       "stops. The lesson: be flexible about HOW you make money and rigid about how you protect it."),
      ("Walton, Lescarbeau, Cook, Galante, Shaw", "A momentum trader who walked away at the top (Walton); a "
       "systematic fund-timer (Lescarbeau); an options tape-reader who built his own 'cumulative tick' (Cook); one "
       "of the few dedicated short-sellers profiled (Galante); and a quant pioneer (D. E. Shaw). Opposite styles, "
       "same backbone: a measured edge, ruthless loss-cutting, and self-knowledge.")]),
    ("4. Hedge Fund Market Wizards (2012)",
     "The institutional generation, interviewed in the long shadow of 2008. Bigger capital and more math -- the "
     "same DNA underneath.",
     [("Ray Dalio", "The founder of Bridgewater, the world's largest hedge fund -- who was once so wrong he had to "
       "lay off everyone and borrow $4,000 from his father to survive. Out of that humiliation came radical "
       "open-mindedness, written 'principles,' and his 'Holy Grail': fifteen or so good, UNCORRELATED return "
       "streams cut risk roughly fourfold without cutting return. Balance over prediction; truth over ego."),
      ("Ed Thorp", "The professor who first beat blackjack with card counting ('Beat the Dealer'), then beat Wall "
       "Street: his Princeton/Newport Partners ran for decades without a losing year, pricing warrants with a "
       "Black-Scholes-style formula before Black-Scholes was published. He sized with the mathematics of "
       "risk-of-ruin (Kelly), and -- famously -- sniffed out Madoff as a fraud years early. Find a real edge; "
       "measure it; bet it carefully."),
      ("Colm O'Shea", "A global-macro trader whose creed is flexibility: 'strong opinions, weakly held.' He will "
       "argue his thesis passionately and abandon it the instant price says otherwise, because 'the market is "
       "always right' and being stubborn is how macro traders die."),
      ("Jamie Mai & Jaffray Woodriff", "Mai (of Cornwall Capital, later immortalized in 'The Big Short') hunts "
       "asymmetric bets -- risk a little on mispriced options to make a lot. Woodriff is a systematic data-miner "
       "who is paranoid about overfitting: his entire process is built around brutal out-of-sample validation, "
       "because a backtest that has not survived genuinely unseen data is just a story.")]),
    ("5. The Little Book of Market Wizards (2014)",
     "Schwager steps back and distills forty-plus interviews into the core principles, stripped of the war stories "
     "-- the single best one-sitting summary of everything the series teaches. It is organized around the lessons "
     "themselves: the primacy of risk, the necessity of a method that fits you, discipline, flexibility, patience, "
     "and the long, unglamorous work of mastering your own psychology.",
     [("Why it matters", "If the other five books are the evidence, this is the verdict. It states plainly what the "
       "interviews keep proving: the market does not reward intelligence or hard work directly -- it rewards "
       "disciplined risk-taking inside an edge you actually have, executed by someone who has made peace with "
       "being wrong.")]),
    ("6. Unknown Market Wizards (2020)",
     "Schwager's return to the roots, three decades on: not fund managers this time, but individual, self-funded "
     "traders with records that shame most institutions -- proof that the principles, not the resources, are what "
     "matter.",
     [("Peter Brandt", "A classical chartist who has traded for over four decades with disciplined, mid-double-digit "
       "compounded returns and tiny risk per trade. 'I am a risk manager first and a trader second.' He trades "
       "well-defined chart patterns, sizes small, and survives the long losing stretches that wash out everyone "
       "chasing the perfect entry."),
      ("Jason Shapiro", "A contrarian by construction: he uses positioning data (the Commitments of Traders) to "
       "find where the crowd is dangerously one-sided, and fades it. His edge is temperament -- the willingness to "
       "be uncomfortable and alone in a trade when the evidence says the consensus is trapped."),
      ("Amrit Sall, Richard Bargh, Daljit Dhaliwal", "A cluster of event/news traders whose careers are built on a "
       "handful of enormous, asymmetric trades a year. They wait, do nothing, and then press ferociously when a "
       "scheduled catalyst lines up with extreme reward-to-risk -- and they obsess over mindset, journaling their "
       "psychology as carefully as their P&L. Sall's insight is essentially the optionality of patience: a few "
       "huge, well-chosen trades make the year; the rest is discipline and waiting."),
      ("The meta-lesson", "None of them have a secret indicator. Their extraordinary returns came from process, "
       "savage risk control, a niche that fit their temperament, and emotional mastery. The 'unknown' wizards prove "
       "the famous ones were never the point -- the principles are.")]),
]

UNIVERSAL = [
    ("Risk management is priority #1", "In every book, capital preservation outranks profit. The wizards think first "
     "about how much they can lose. 'Play great defense, not great offense' is the closest thing the series has to "
     "a creed."),
    ("Cut losses short -- always", "Universal and non-negotiable. Losses must stay small; a single large loss can "
     "undo a year of good trades and, worse, your confidence. Seykota's triple repetition is not a joke."),
    ("Size for survival", "Risk a small, fixed fraction per trade -- commonly ~1-2%, sometimes far less. Kovner's "
     "'undertrade' and Hite's 'if you lose all your chips you can't bet' are the same rule. You cannot win once you "
     "are out of the game."),
    ("Let winners run", "Payoffs are asymmetric: a few large wins pay for a long tail of small losses. Taking "
     "profits early FEELS responsible and quietly guarantees mediocrity (Eckhardt's warning)."),
    ("Find a method that fits YOU", "There is no one right way -- the book contains trend-followers and contrarians, "
     "quants and chartists, all rich. The method must match your personality, beliefs, and life, or you will not "
     "follow it under stress."),
    ("The edge is in execution", "A good idea without discipline is worthless. The repeatable money comes from "
     "consistent, rule-bound execution -- the same trade, taken the same way, a thousand times."),
    ("Master your psychology", "You are your own worst enemy. Ego, hope, fear, and impatience cost more than any bad "
     "forecast. Schwartz's breakthrough -- divorcing ego from the trade -- recurs in nearly every interview."),
    ("Have a defined, tested edge", "Know WHY you win: a method with measured positive expectancy, not a hunch. "
     "Woodriff's out-of-sample paranoia is the institutional version of the same instinct."),
    ("Think independently", "The wizards do not follow the crowd; several (Shapiro) fade it for a living. Conviction "
     "comes from your own analysis, not consensus or tips -- Marcus's early ruin came from a tip."),
    ("Adapt -- edges decay", "Markets change and methods stop working (the whole lesson of Stock Market Wizards). "
     "The greats evolve the method continuously while keeping the risk rules fixed."),
    ("Love the process, not the money", "Durable performance comes from genuine passion for the craft; money is the "
     "by-product. The ones who chased money burned out or blew up."),
    ("Keep records; learn from mistakes", "Journaling and honest post-mortems compound skill. The Unknown Wizards "
     "journal their psychology as rigorously as their trades. Mistakes are tuition -- only if you study them."),
    ("Strong opinions, weakly held", "Conviction to act, flexibility to reverse the instant the thesis breaks. "
     "'The market is always right' (O'Shea). Stubbornness is how good traders die."),
    ("Patience -- wait for your pitch", "Sitting on your hands is a position. The best returns come from a few "
     "high-quality, asymmetric setups (Sall, Lipschutz), not constant activity. Most traders lose by overtrading."),
]

QUOTES = [
    ("The elements of good trading are cutting losses, cutting losses, and cutting losses.", "Ed Seykota"),
    ("Everybody gets what they want out of the markets.", "Ed Seykota"),
    ("Don't be a hero. Don't have an ego.", "Paul Tudor Jones"),
    ("I'm always thinking about losing money as opposed to making money.", "Paul Tudor Jones"),
    ("Undertrade, undertrade, undertrade.", "Bruce Kovner"),
    ("What feels good is often the wrong thing to do.", "William Eckhardt"),
    ("If most traders would learn to sit on their hands 50 percent of the time, they would make a lot more money.",
     "Bill Lipschutz"),
    ("The most important change in my trading career was when I learned to divorce my ego from the trade.",
     "Marty Schwartz"),
    ("If you don't bet, you can't win. If you lose all your chips, you can't bet.", "Larry Hite"),
    ("I am a risk manager first and a trader second.", "Peter Brandt"),
    ("The market is always right.", "Colm O'Shea (and the macro wizards)"),
]


def build():
    pdf = PDF(format="A4"); pdf.set_auto_page_break(True, margin=16); pdf.set_margins(18, 16, 18)

    pdf.add_page(); pdf.ln(38)
    pdf.set_font("Helvetica", "B", 26); pdf.set_text_color(*ACCENT)
    pdf.multi_cell(0, 12, "The Market Wizards Series", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 15); pdf.set_text_color(*INK)
    pdf.multi_cell(0, 8.5, "Stories & Distilled Lessons from All Six Books", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6); pdf.set_font("Helvetica", "", 12); pdf.set_text_color(*GREY)
    pdf.multi_cell(0, 6, _ascii("Jack D. Schwager interviewed the world's greatest traders across three decades "
                                "(1989-2020). This is the long-form summary: the defining stories behind each "
                                "wizard, then the universal principles that recur in every interview, market, and "
                                "era."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8); pdf.set_font("Helvetica", "I", 9.5)
    pdf.multi_cell(0, 5, _ascii("A study distillation. Anecdotes and quotes render the widely-cited substance of "
                                "each trader's account; figures are as reported in the books and are approximate. "
                                "Compiled 2026-06-06."), new_x="LMARGIN", new_y="NEXT")

    pdf.add_page(); pdf.h1("Preface")
    pdf.para("Over six books and thirty years, Schwager interviewed the best traders alive -- in futures, FX, "
             "equities, options, global macro, quant, and the ranks of self-funded individuals. They trade opposite "
             "styles on opposite timeframes: trend-followers and contrarians, discretionary tape-readers and "
             "systematic data-miners. And yet, read end to end, the series is almost monotonous in its agreement. "
             "The methods are personal and infinitely various; the principles are few and universal. What follows "
             "tells enough of each story to make the lessons stick, then collects the lessons themselves.")

    pdf.h1("Part I -- The Six Books, and the Wizards In Them")
    for title, blurb, people in BOOKS:
        pdf.h2(title); pdf.para(blurb, size=10.0)
        for who, story in people:
            pdf.name(who); pdf.para(story)
        pdf.ln(0.5)

    pdf.add_page(); pdf.h1("Part II -- The Universal Lessons")
    pdf.para("These appear in every book. If the series has a single thesis it is this: the method is personal and "
             "negotiable; the risk discipline and the self-mastery are universal and not.")
    for i, (head, body) in enumerate(UNIVERSAL, 1):
        pdf.set_x(pdf.l_margin); pdf.set_font("Helvetica", "B", 11)
        pdf.multi_cell(0, 5.6, _ascii(f"{i}. {head}"), new_x="LMARGIN", new_y="NEXT"); pdf.para(body)

    pdf.add_page(); pdf.h1("Part III -- Memorable Lines")
    for t, who in QUOTES:
        pdf.quote(t, who)
    pdf.ln(3); pdf.h1("Closing")
    pdf.para("The wizards are not unified by a secret indicator or a magic market. They are unified by defense: "
             "protect capital, cut losses, size for survival, press only your best ideas, adapt as the world "
             "changes, and -- hardest of all -- master yourself. The edge that lasts is not a setup; it is a "
             "temperament, applied with discipline. Everything else is style.")

    out = "deliverables/Market_Wizards_Distilled.pdf"; pdf.output(out); return out


if __name__ == "__main__":
    print("wrote", build())
