"""Every import inside src/ must point at code that exists in this repo (static check: nothing is imported).

Code ported from the original research repos kept imports that only worked there — `from utils import wiki_logger`
(a module never ported), `from misc.cleaning import …` (the module lives under dataset.wikipedia.misc),
`from benchmarks.prediction import …` (it lives under benchmarks/wrappers/). Each made a whole pipeline step fail at
import time, unnoticed because no test imported it. Rules (AST over every .py under src/):
  1. an import of one of our top-level packages (dataset, evaluation, …) must resolve to an existing module;
  2. a relative import must resolve inside src/;
  3. a bare first name that is also the name of a NESTED repo module (e.g. `utils`, `prediction`) is allowed only
     when that module is in the importing file's own folder or an ancestor folder (script-style code run from its
     folder, e.g. the benchmark wrappers whose run.sh put those folders on PYTHONPATH);
  4. installed packages that merely share a name with a local folder are listed explicitly (INSTALLED_NAMESAKES);
  5. for script-style imports, the PYTHONPATH folders declared by the nearest run.sh count too (full module path).
"""
import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / 'src'
# installed third-party packages whose name also exists as a local folder/module (not repo-internal imports)
INSTALLED_NAMESAKES = {'relik', 'unified_llm_client'}


def _exists(base: Path, dotted: str) -> bool:
    target = base.joinpath(*dotted.split('.'))
    return target.with_suffix('.py').is_file() or target.is_dir()


def _run_sh_paths(folder: Path, src: Path) -> list:
    """PYTHONPATH folders declared by the nearest run.sh (the wrappers' launcher), e.g.
    `export PYTHONPATH="${SCRIPT_DIR}/../edc_plus/edc_tt2kg:${SCRIPT_DIR}/..:${PYTHONPATH:-}"`."""
    for d in [folder, *folder.parents]:
        if d != src and src not in d.parents:
            break
        run_sh = d / 'run.sh'
        if run_sh.is_file():
            paths = []
            for line in run_sh.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if line.startswith('export PYTHONPATH='):
                    for entry in line.split('=', 1)[1].strip('"\'').split(':'):
                        if '${SCRIPT_DIR}' in entry:
                            paths.append((d / entry.replace('${SCRIPT_DIR}', '').lstrip('/')).resolve())
            return paths
    return []


def find_unresolvable_imports(src: Path) -> list:
    files = sorted(p for p in src.rglob('*.py') if '__pycache__' not in p.parts)
    top_level = {p.name for p in src.iterdir() if p.is_dir() and p.name != '__pycache__'} | \
                {p.stem for p in src.glob('*.py')}
    nested = {p.stem for p in files if len(p.relative_to(src).parts) > 1 and p.stem != '__init__'} | \
             {p.name for p in src.rglob('*') if p.is_dir() and p.name != '__pycache__' and p.parent != src}
    problems = []
    for path in files:
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [(alias.name, 0) for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [(node.module or '', node.level)]
            else:
                continue
            where = f'{path.relative_to(src.parent)}:{node.lineno}'
            for module, level in modules:
                if level > 0:                                                      # rule 2
                    base = path.parent
                    for _ in range(level - 1):
                        base = base.parent
                    if module and not _exists(base, module):
                        problems.append(f'{where}: relative import {"." * level}{module} does not resolve')
                    continue
                root = module.split('.')[0]
                if root in top_level:                                              # rule 1
                    if not _exists(src, module):
                        problems.append(f'{where}: {module} does not exist under src/')
                elif root in nested and root not in INSTALLED_NAMESAKES:           # rules 3, 4
                    ancestors = [d for d in [path.parent, *path.parent.parents] if d == src or src in d.parents]
                    search = ancestors + _run_sh_paths(path.parent, src)
                    if not any(_exists(d, module) for d in search):
                        problems.append(f'{where}: `{module}` looks like a repo-internal import that only worked '
                                        f'in the original repo (not found in this file\'s folder, its parents or '
                                        f'its run.sh PYTHONPATH)')
    return problems


def test_every_import_in_src_resolves():
    problems = find_unresolvable_imports(SRC)
    assert not problems, 'imports that cannot work in this repo:\n' + '\n'.join(problems)


def test_the_checker_catches_the_bugs_it_was_written_for(tmp_path):
    """A tiny fake src/ with the three historical bug patterns, one missing bundled module and four valid
    imports (package path, relative, script-style, run.sh PYTHONPATH): exactly 4 reported."""
    src = tmp_path / 'src'
    for d in ('dataset/wikipedia/misc', 'dataset/emerge/utils', 'misc', 'benchmarks/wrappers/relik'):
        (src / d).mkdir(parents=True)
    for f in ('dataset/__init__.py', 'dataset/wikipedia/__init__.py', 'dataset/wikipedia/misc/__init__.py',
              'dataset/wikipedia/misc/cleaning.py', 'dataset/emerge/utils/wiki_utils.py', 'misc/__init__.py',
              'benchmarks/wrappers/prediction.py'):
        (src / f).write_text('')
    (src / 'dataset/wikipedia/misc/extractor.py').write_text(
        'from utils import wiki_logger\n'                       # never-ported module            -> reported
        'from misc.cleaning import x\n'                         # wrong package path             -> reported
        'from dataset.wikipedia.misc.cleaning import y\n'       # correct                        -> ok
        'from .cleaning import z\n')                            # correct relative               -> ok
    (src / 'benchmarks/wrappers/relik/model.py').write_text(
        'from benchmarks.prediction import P\n'                 # module is in wrappers/         -> reported
        'from prediction import P\n'                            # script-style, ancestor folder  -> ok
        'from relik import Relik\n')                            # installed namesake             -> ok
    # a wrapper whose run.sh puts a bundled tree on PYTHONPATH (the KG-aware wrapper uses edc_plus/edc_tt2kg)
    (src / 'benchmarks/wrappers/kg_prompt').mkdir()
    (src / 'benchmarks/wrappers/edc_plus/edc_tt2kg/edc/utils').mkdir(parents=True)
    (src / 'benchmarks/wrappers/edc_plus/edc_tt2kg/edc/utils/client.py').write_text('')
    (src / 'benchmarks/wrappers/kg_prompt/run.sh').write_text(
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'export PYTHONPATH="${SCRIPT_DIR}/../edc_plus/edc_tt2kg:${SCRIPT_DIR}/..:${PYTHONPATH:-}"\n')
    (src / 'benchmarks/wrappers/kg_prompt/wrapper.py').write_text(
        'from edc.utils.client import C\n'                      # via run.sh PYTHONPATH          -> ok
        'from edc.utils.missing import M\n')                    # not in the bundled tree        -> reported
    found = find_unresolvable_imports(src)
    assert len(found) == 4, found
    assert any('extractor.py:1' in p and 'utils' in p for p in found)
    assert any('extractor.py:2' in p and 'misc.cleaning' in p for p in found)
    assert any('model.py:1' in p and 'benchmarks.prediction' in p for p in found)
    assert any('kg_prompt/wrapper.py:2' in p and 'edc.utils.missing' in p for p in found)
