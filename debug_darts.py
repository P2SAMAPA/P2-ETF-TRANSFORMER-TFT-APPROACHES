import darts
print('darts version:', darts.__version__)
print('darts path:', darts.__file__)
import os
forecasting_path = os.path.join(os.path.dirname(darts.__file__), 'models', 'forecasting')
print('Contents of darts/models/forecasting:')
if os.path.exists(forecasting_path):
    for f in os.listdir(forecasting_path):
        if f.endswith('.py') or f.endswith('.pyc'):
            print(' -', f)
else:
    print('Path not found:', forecasting_path)

print('\nTrying imports:')
try:
    from darts.models.forecasting.tft_model import TFTModel
    print('✅ TFTModel imported from tft_model')
except Exception as e:
    print('❌ TFTModel import failed:', e)

try:
    from darts.models.forecasting.patchtst_model import PatchTSTModel
    print('✅ PatchTSTModel imported from patchtst_model')
except Exception as e:
    print('❌ PatchTSTModel import from patchtst_model failed:', e)

try:
    from darts.models.forecasting.patchtst import PatchTSTModel
    print('✅ PatchTSTModel imported from patchtst')
except Exception as e:
    print('❌ PatchTSTModel import from patchtst failed:', e)

try:
    from darts.models import PatchTSTModel
    print('✅ PatchTSTModel imported from darts.models')
except Exception as e:
    print('❌ PatchTSTModel import from darts.models failed:', e)
