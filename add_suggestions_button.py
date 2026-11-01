#!/usr/bin/env python3
"""add_suggestions_button.py — put the '+ Suggestions' button next to '+ Veo import' in web/components/GameView.tsx (in place).

Run from the repository root:  python3 add_suggestions_button.py     (safe to run twice)
Also works on web/app/game/[id]/page.tsx if that is where your Veo button lives: pass the path as an argument.
"""
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "web/components/GameView.tsx"
s = open(path).read()
if "SuggestionImport" in s:
    print(path, "already has the suggestions button: nothing to do")
    sys.exit(0)

# 1) the import
m = re.search(r"^import VeoImportModal from '@/components/VeoImportModal'\n", s, re.M)
if not m:
    sys.exit("could not find the VeoImportModal import line: add\n  import SuggestionImport from '@/components/SuggestionImport'\nby hand")
s = s[:m.end()] + "import SuggestionImport from '@/components/SuggestionImport'\n" + s[m.end():]

# 2) the button, right after the '+ Veo import' button block
m = re.search(r"\+ Veo import\s*</button>\s*\)\}\n", s)
if not m:
    sys.exit("could not find the '+ Veo import' button in " + path + ": tell me and I will make the edit for your file")
el = """                  {isCoach && userId && (
                    <SuggestionImport
                      gameId={id}
                      userId={userId}
                      shift={boardState.status === 'ready' ? boardState.board.shift : 0}
                      existing={annotations}
                      homeTeam={game.home_team}
                      awayTeam={game.away_team}
                      onImported={() =>
                        supabase
                          .from('annotations')
                          .select('*, profiles(display_name)')
                          .eq('game_id', id)
                          .order('timestamp_sec')
                          .then(({ data }) => { if (data) setAnnotations(data) })
                      }
                    />
                  )}
"""
s = s[:m.end()] + el + s[m.end():]
open(path, "w").write(s)
print(path, "updated: import + button")
