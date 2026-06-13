import pathlib
import runpy

if __name__ == '__main__':
    scripts_dir = pathlib.Path(__file__).parent
    for script in [
        'gen_octant_images.py',
        'gen_vmaps.py',
    ]:
        runpy.run_path(
            str(scripts_dir / script),
            run_name='__main__',
        )
