
from pythologist_reader.formats.inform.sets import CellSampleInForm, CellProjectInForm
from pythologist_reader.formats.inform.custom import CellSampleInFormCustomMask, CellProjectInFormCustomMask
from pythologist_reader.formats.inform.custom import CellSampleInFormLineArea, CellProjectInFormLineArea
import os
from tempfile import NamedTemporaryFile
def read_standard_format_project(path,
	                             annotation_strategy,
	                             project_name='myproject',
	                             temp_dir=None,
	                             custom_mask_name=None,
	                             other_mask_name=None,
	                             microns_per_pixel=0.496,
	                             verbose=False
	                             ):
    """
    Read a project structed in a standardized format starting from the fold containing a list of samples

    Return a dictinary of CellProjects keyed to the export names
    """
    _check_annotation_strategy(annotation_strategy)
    if temp_dir is not None and not os.path.exists(temp_dir):
    	os.makedirs(temp_dir)

    # Get the export names
    samples = [x for x in os.listdir(path) if x[0]!='.']
    if len(samples) == 0: 
        return None

    for sample in samples:
        sample_folder = os.path.join(path,sample)
        exports = [x for x in os.listdir(os.path.join(sample_folder,'INFORM_ANALYSIS')) if x[0]!='.']
        break

    outputs = {}

    # prepare the exports
    for export in exports:
        temp_path = NamedTemporaryFile(prefix='project-',suffix='.h5',dir=temp_dir,delete=False)
        print(temp_path.name)
        if annotation_strategy=='GIMP_CUSTOM':
            cpi = CellProjectInFormCustomMask(temp_path.name,mode='w')
        elif annotation_strategy=='GIMP_TSI':
            cpi = CellProjectInFormLineArea(temp_path.name,mode='w')
        for sample in samples:
            sample_folder = os.path.join(path,sample)
            csi = CellSampleInFormCustomMask()
            csi.read_path(
                      path=os.path.join(sample_folder,'INFORM_ANALYSIS',export),
                      sample_name=sample_folder,
                      verbose = verbose,
                      require = False,
                      require_score = False,
                      custom_mask_name = custom_mask_name,
                      other_mask_name = other_mask_name,
                      alternate_annotation_path = os.path.join(sample_folder,'ANNOTATIONS')
                     )
            cpi.append_sample(csi)
        outputs[export] = cpi
        cpi.project_name = export
        cpi.microns_per_pixel = microns_per_pixel


    return outputs

def read_standard_sample(path, annotation_strategy):
    """
    Read a sample structed in a standardized format.

    Return a dictinary of CellProjects keyed to the export names
    """
    _check_annotation_strategy(annotation_strategy)


    return



def _check_annotation_strategy(annotation_strategy):
    if annotation_strategy not in ['GIMP_TSI','GIMP_CUSTOM','INFORM_ANALYSIS','NO_ANNOTATION']:
        raise ValueError("Declare the annotation strategy.")
