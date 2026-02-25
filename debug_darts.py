import darts
import inspect
import os

print('darts version:', darts.__version__)
print('darts path:', darts.__file__)

# List files in forecasting directory
forecasting_path = os.path.join(os.path.dirname(darts.__file__), 'models', 'forecasting')
print('\nContents of darts/models/forecasting:')
if os.path.exists(forecasting_path):
    for f in sorted(os.listdir(forecasting_path)):
        if f.endswith('.py') and f != '__init__.py':
            print(' -', f)
else:
    print('Path not found:', forecasting_path)

# Now try to list all classes available in darts.models.forecasting
print('\nClasses in darts.models.forecasting:')
try:
    from darts.models import forecasting
    forecasting_classes = [name for name, obj in inspect.getmembers(forecasting) if inspect.isclass(obj)]
    for cls in sorted(forecasting_classes):
        print(' -', cls)
except Exception as e:
    print('Could not inspect forecasting module:', e)

# Specifically look for any class with "patch" in the name (case-insensitive)
print('\nSearching for any class containing "patch" (case-insensitive):')
try:
    from darts.models import forecasting
    patch_classes = [name for name, obj in inspect.getmembers(forecasting) if inspect.isclass(obj) and 'patch' in name.lower()]
    if patch_classes:
        for cls in patch_classes:
            print(' -', cls)
    else:
        print('No class with "patch" found.')
except Exception as e:
    print('Error searching:', e)

# Also try to import from darts.models directly
print('\nTrying direct import from darts.models:')
try:
    from darts.models import PatchTSTModel
    print('✅ PatchTSTModel imported from darts.models')
except Exception as e:
    print('❌ Failed:', e)

# Try alternative imports
print('\nTrying alternative imports:')
alternative_paths = [
    'darts.models.forecasting.patchtst_model',
    'darts.models.forecasting.patchtst',
    'darts.models.forecasting.patch_tst',
    'darts.models.forecasting.patch_tst_model',
]

for mod_path in alternative_paths:
    try:
        mod = __import__(mod_path, fromlist=['PatchTSTModel'])
        if hasattr(mod, 'PatchTSTModel'):
            print(f'✅ Found PatchTSTModel in {mod_path}')
        else:
            print(f'❌ Module {mod_path} exists but no PatchTSTModel')
    except ImportError:
        print(f'❌ Cannot import {mod_path}')
