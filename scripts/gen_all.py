import pathlib
import runpy
import subprocess

if __name__ == '__main__':
    scripts_dir = pathlib.Path(__file__).parent
    julia_bin = (
        pathlib.Path.home()
        / '.juliaup'
        / 'bin'
        / 'julia'
    )
    for script in [
        'gen_octant_images.py',
        'gen_octant_shapes.py',
        'gen_vmaps.py',
        'gen_sfrs.jl',
        'gen_firebox_summary_stats.jl',
    ]:
        script_path = scripts_dir / script
        if script.endswith('.py'):
            runpy.run_path(
                str(script_path),
                run_name='__main__',
            )
        elif script.endswith('.jl'):
            subprocess.run(
                [
                    str(julia_bin),
                    '--project=' + str(scripts_dir),
                    str(script_path),
                ],
                check=True,
            )
