import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import shutil

import os
import sys
import shlex, subprocess
import config
from tools import general_tools as gt


def save_transform_and_image(transform, fixed_image, moving_image, output_path, outputfile_prefix):
    """
    Write the given transformation to file, resample the moving_image onto the fixed_images grid and save the
    result to file.

    Args:
        transform (SimpleITK Transform): transform that maps points from the fixed image coordinate system to the moving.
        fixed_image (SimpleITK Image): resample onto the spatial grid defined by this image.
        moving_image (SimpleITK Image): resample this image.
        outputfile_prefix (string): transform is written to outputfile_prefix.tfm and resampled image is written to
                                    outputfile_prefix.mha.
    """
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(fixed_image)

    # SimpleITK supports several interpolation options, we go with the simplest that gives reasonable results.
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetTransform(transform)
    image_resampled = resample.Execute(moving_image)
    #print(os.path.join(output_path, outputfile_prefix))
    sitk.WriteImage(image_resampled, os.path.join(output_path, outputfile_prefix + '.nii'))
    sitk.WriteTransform(transform, os.path.join(output_path, outputfile_prefix + '.tfm'))
    return image_resampled




def display_images(fixed_image_z, moving_image_z, fixed_npa, moving_npa):
    # Create a figure with two subplots and the specified size.
    plt.subplots(1, 2, figsize=(10, 8))

    # Draw the fixed image in the first subplot.
    plt.subplot(1, 2, 1)
    plt.imshow(fixed_npa[fixed_image_z, :, :], cmap=plt.cm.Greys_r);
    plt.title('fixed image')
    plt.axis('off')

    # Draw the moving image in the second subplot.
    plt.subplot(1, 2, 2)
    plt.imshow(moving_npa[moving_image_z, :, :], cmap=plt.cm.Greys_r);
    plt.title('moving image')
    plt.axis('off')

    plt.show()

# Callback invoked by the IPython interact method for scrolling and modifying the alpha blending
# of an image stack of two images that occupy the same physical space.
def display_images_with_alpha(image_z, alpha, fixed, moving):
    img = (1.0 - alpha)*fixed[:,:,image_z] + alpha*moving[:,:,image_z]
    plt.imshow(sitk.GetArrayViewFromImage(img),cmap=plt.cm.Greys_r);
    plt.axis('off')
    plt.show()


# Callback invoked when the StartEvent happens, sets up our new data.
def start_plot():
    global metric_values, multires_iterations

    metric_values = []
    multires_iterations = []


# Callback invoked when the EndEvent happens, do cleanup of data and figure.
def end_plot():
    global metric_values, multires_iterations

    del metric_values
    del multires_iterations
    # Close figure, we don't want to get a duplicate of the plot latter on.
    plt.close()


# Callback invoked when the IterationEvent happens, update our data and display new figure.
def plot_values(registration_method):
    global metric_values, multires_iterations

    metric_values.append(registration_method.GetMetricValue())
    # Clear the output area (wait=True, to reduce flickering), and plot current data
    #clear_output(wait=True)
    # Plot the similarity metric values
    plt.plot(metric_values, 'r')
    plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
    plt.xlabel('Iteration Number', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.show()


# Callback invoked when the sitkMultiResolutionIterationEvent happens, update the index into the
# metric_values list.
def update_multires_iterations():
    global metric_values, multires_iterations
    multires_iterations.append(len(metric_values))


def register_simple(fixed_image, moving_image):
    initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                          moving_image,
                                                          sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.MOMENTS)

    # moving_resampled = sitk.Resample(moving_image, fixed_image, initial_transform, sitk.sitkLinear, 0.0,
    #                                  moving_image.GetPixelID())
    # display_images(10, 10, sitk.GetArrayFromImage(fixed_image), sitk.GetArrayFromImage(moving_image))

    registration_method = sitk.ImageRegistrationMethod()

    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    registration_method.SetInterpolator(sitk.sitkLinear)

    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
    # Scale the step size differently for each parameter, this is critical!!!
    registration_method.SetOptimizerScalesFromPhysicalShift()

    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    #registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
    #registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
    #registration_method.AddCommand(sitk.sitkIterationEvent,
    #                               lambda: plot_values(registration_method))

    final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
                                                  sitk.Cast(moving_image, sitk.sitkFloat32))
    return final_transform




def coarse_alignment(fixed_image, moving_image, output_path, output_prefix="registration"):
    # == INITIAL TRANSFORM
    initial_transform = sitk.CenteredTransformInitializer(sitk.Cast(fixed_image, moving_image.GetPixelID()),
                                                          moving_image,
                                                          sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registered_image = save_transform_and_image(initial_transform, fixed_image, moving_image,
                                                output_path, os.path.join(output_prefix + '_initial'))
    return initial_transform


def register(fixed_image, moving_image, output_path, registration_type='rigid', output_prefix="registration", fixed_mask=None):
    """
    fixed_mask: only points WITHIN the mask will we used for registration, from example '65_Registration_FFD.ipynb'
    """
    # == INITIAL TRANSFORM
    initial_transform = coarse_alignment(fixed_image, moving_image, output_path, output_prefix)

    # == MAIN REGISTRATION
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    if fixed_mask:
        registration_method.SetMetricFixedMask(fixed_mask)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0,
                                                      numberOfIterations=100)  # , estimateLearningRate=registration_method.EachIteration)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    #== Choose Registration Type
    if registration_type=='rigid':
        final_transform = sitk.Euler3DTransform(initial_transform)
    elif registration_type=='affine':
        final_transform = sitk.AffineTransform(3)
        final_transform.SetTranslation(initial_transform.GetParameters()[3:])
        final_transform.SetCenter(initial_transform.GetFixedParameters()[0:3])
    elif registration_type=='demon':
        # Create initial identity transformation.
        transform_to_displacement_field_filter = sitk.TransformToDisplacementFieldFilter()
        transform_to_displacement_field_filter.SetReferenceImage(fixed_image)
        # The image returned from the initial_transform_filter is transferred to the transform and cleared out.
        final_transform = sitk.DisplacementFieldTransform(
            transform_to_displacement_field_filter.Execute(initial_transform))
    else:
        print("Registration type '%s' not defined ... exiting."%(registration_type))
        return 0

    #== INITIALIZE AND RUN REGISTRATION
    registration_method.SetInitialTransform(final_transform)
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[8, 4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[3, 2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
                                sitk.Cast(moving_image, sitk.sitkFloat32))

    #== OUTPUT
    print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
    print('Final metric value: {0}'.format(registration_method.GetMetricValue()))

    registered_image = save_transform_and_image(final_transform, fixed_image, moving_image,
                                                output_path, os.path.join(output_prefix + '_final'))
    return registered_image






def build_ants_registration_command(fixed_img, moving_img, output_prefix, registration_type='Rigid', image_ext='mha',
                                    fixed_mask=None, moving_mask=None, verbose=0):
    if registration_type=='Syn':
        reg_params = [0.1, 3, 0]
        metric = 'CC'+str([fixed_img, moving_img, 1, 4])
        convergence = '[100x70x50x20,1e-6,10]'
    else:
        reg_params = [0.1]
        metric = 'MI'+str([fixed_img, moving_img, 1, 32, 'Regular', 0.25])
        convergence = '[1000x500x250x100,1e-6,10]'
    ants_params_dict = { 'verbose'        : verbose,
                         'dimensionality' : 3,
                         'output'         : [ output_prefix, output_prefix+'.'+image_ext ],
                         'interpolation'  : 'Linear',
                         'winsorize-image-intensities': [ 0.005 , 0.995 ],
                         'use-histogram-matching':       1,
                         'initial-moving-transform':     [fixed_img, moving_img, 1],
                         'transform'      : registration_type+str(reg_params),
                         'metric'         :  metric,
                         'convergence'    :  convergence,
                         'shrink-factors' : '8x4x2x1',
                         'smoothing-sigmas': '3x2x1x0vox'}
    if fixed_mask and moving_mask:
        ants_params_dict['x']='['+fixed_mask+','+moving_mask+']'
    elif fixed_mask:
        ants_params_dict['x'] = fixed_mask
    elif moving_mask:
        ants_params_dict['x'] = '[,'+moving_mask+']'
    ants_params_str = ' --'.join([' '.join([key, str(value)]) for key, value in ants_params_dict.items()])
    ants_cmd = 'antsRegistration --'+ants_params_str
    return ants_cmd


def register_ants(fixed_img, moving_img, output_prefix, path_to_transform=None, registration_type='Rigid',
                  image_ext='mha', fixed_mask=None, moving_mask=None, verbose=0):
    print("  - Starting ANTS registration:")
    print("    - FIXED IMG : %s"%fixed_img)
    print("    - MOVING IMG: %s" % moving_img)
    print("    - OUTPUT    : %s" % output_prefix)
    ants_cmd = build_ants_registration_command(fixed_img, moving_img, output_prefix, registration_type, image_ext,
                                               fixed_mask, moving_mask, verbose)
    print("ANTS command: %s"%ants_cmd)
    gt.ensure_dir_exists(os.path.dirname(output_prefix))
    args = shlex.split(ants_cmd)
    process = subprocess.Popen(args)
    process.wait()
    if process.returncode==0 and path_to_transform!=None:
        #-- rename trafo file
        if registration_type=='Rigid' or registration_type=='Affine':
            path_to_transform_ants = output_prefix+'0GenericAffine.mat'
            shutil.move(path_to_transform_ants, path_to_transform)
        if registration_type=='Syn':
            path_to_transform_ants = output_prefix + '1Warp.nii.gz'
            shutil.move(path_to_transform_ants, path_to_transform)
    print("Registration terminated with return code: '%s'"%process.returncode)

    return process.returncode


def register_ants_synquick(fixed_img, moving_img, output_prefix, registration='s', fixed_mask=None ):
    """
    registration:
        - r -> rigid
        - a -> rigid, affine
        - s -> rigid, affine, syn
    """
    ants_script_path = config.ants_script_path
    ants_params_dict = {'d' : 3,
                        'f': fixed_img,
                        'm': moving_img,
                        't': registration,
                        'o': output_prefix,
                        'n': 4,
                        'j': 1,
                        'z': 0 }
    if fixed_mask:
        ants_params_dict['x'] = fixed_mask
    ants_params_str = ' -'.join([' '.join([key, str(value)]) for key, value in ants_params_dict.items()])
    #ants_cmd = "%s -"%(os.path.join(ants_script_path, 'antsRegistrationSyNQuick.sh')) + ants_params_str
    ants_cmd = "%s -"%(os.path.join(ants_script_path, 'antsRegistrationSyNQuick.sh')) + ants_params_str
    print("ANTS command SYNquick: %s" % ants_cmd)
    gt.ensure_dir_exists(os.path.dirname(output_prefix))
    args = shlex.split(ants_cmd)
    process = subprocess.Popen(args, env=config.path_environment)
    process.wait()
    print("ANTS terminated with return code: '%s'" % process.returncode)


def segmentation_to_mask(path_to_seg_in, path_to_seg_out, merge=None, invert=True):
    """
    -merge expecteds tuple [label_value_min, label_value_max, target_lable_value]
    """
    path_to_seg_current = path_to_seg_in
    if merge:
        path_to_seg_out_merged = "%s_%s.%s"%(path_to_seg_out.split('.')[0], 'mergedlabels',
                                            gt.get_file_extension(path_to_seg_out))
        ants_merge_labels_cmd = "ImageMath 3 %s ReplaceVoxelValue %s %s %s %s"%(
                                path_to_seg_out_merged, path_to_seg_current,
                                str(merge[0]), str(merge[1]), str(merge[2]) )
        path_to_seg_current = path_to_seg_out_merged
        print("ANTS command merging: %s" % ants_merge_labels_cmd)
        gt.ensure_dir_exists(os.path.dirname(path_to_seg_out_merged))
        args = shlex.split(ants_merge_labels_cmd)
        process = subprocess.Popen(args)
        process.wait()
        print("ANTS merging terminated with return code: '%s'" % process.returncode)
    if invert:
        path_to_seg_out_invert = "%s_%s.%s"%(path_to_seg_out.split('.')[0], 'inverted',
                                            gt.get_file_extension(path_to_seg_out))
        ants_invert_labels_cmd = "ImageMath 3 %s Neg %s" % (
                                            path_to_seg_out_invert, path_to_seg_current)
        path_to_seg_current = path_to_seg_out_invert
        print("ANTS command invert: %s" % ants_invert_labels_cmd)
        gt.ensure_dir_exists(os.path.dirname(path_to_seg_current))
        args = shlex.split(ants_invert_labels_cmd)
        process = subprocess.Popen(args)
        process.wait()
        print("ANTS invert command terminated with return code: '%s'" % process.returncode)
    shutil.copy(path_to_seg_current, path_to_seg_out)



def build_ants_apply_transforms_command(input_img, reference_img, output_file, transforms):
    ants_params_dict = {'verbose': 0,
                        'dimensionality': 3,
                        'input-image-type': 'scalar',
                        'input' : input_img,
                        'reference-image': reference_img,
                        'output': output_file,
                        'interpolation': 'Linear'}
    ants_params_str = ' --'.join([' '.join([key, str(value)]) for key, value in ants_params_dict.items()])
    for transform in transforms:
        ants_params_str = ants_params_str + ' --transform %s'%transform
    ants_cmd = 'antsApplyTransforms --'+ants_params_str
    return ants_cmd


def ants_apply_transforms(input_img, reference_img, output_file, transforms):
    print("  - Starting ANTS Apply Transforms:")
    print("    - INPUT IMG      : %s"%input_img)
    print("    - REFERENCE IMG  : %s" % reference_img)
    print("    - OUTPUT         : %s" % output_file)
    ants_cmd = build_ants_apply_transforms_command(input_img, reference_img, output_file, transforms)
    print("ANTS command: %s"%ants_cmd)
    gt.ensure_dir_exists(os.path.dirname(output_file))
    args = shlex.split(ants_cmd)
    process = subprocess.Popen(args, env=config.path_environment)
    process.wait()
    #return process.returncode
    print("ANTS apply transforms terminated with return code: '%s'"%process.returncode)


def ants_image_maths(input_img_1, input_img_2, output_file, cmd):
    print("  - Starting ANTS Image Math:")
    print("    - INPUT IMG1  : %s"%input_img_1)
    print("    - INPUT IMG2  : %s" % input_img_2)
    print("    - OUTPUT      : %s" % output_file)
    print("    - COMMAND     : %s" % cmd)
    ants_cmd = "ImageMath 3 %s %s %s %s"%(output_file, cmd, input_img_1, input_img_2)
    print("ANTS command: %s"%ants_cmd)
    gt.ensure_dir_exists(os.path.dirname(output_file))
    args = shlex.split(ants_cmd)
    process = subprocess.Popen(args)
    process.wait()
    #return process.returncode
    print("ANTS ImageMath terminated with return code: '%s'"%process.returncode)


def ants_skull_strip(path_to_img, path_to_atlas, path_to_atlas_brain_mask,
                     ref_mod_name='T1', output_dir=None,
                     image_ext='.mha', fixed_mask=None,
                     also_apply_to=[]):
    if not output_dir:
        output_dir = os.path.dirname(path_to_img)
    #-- temp files
    atlas_reg_rigid_prefix      = os.path.join(output_dir, 'atlas_rigid_reg_to_'+ref_mod_name)
    atlas_reg_affine_prefix     = os.path.join(output_dir, 'atlas_affine_reg_to_'+ref_mod_name)
    rigid_transform             = atlas_reg_rigid_prefix + '0GenericAffine.mat'
    affine_transform            = atlas_reg_affine_prefix + '0GenericAffine.mat'
    path_to_brainmask_reg_affine = os.path.join(output_dir, 'brainmask_' + ref_mod_name + '_affine'+image_ext)
    path_to_skullstripped        = os.path.join(output_dir, ref_mod_name+'_skullstripped'+image_ext)

    print(atlas_reg_rigid_prefix)
    #-- registrations
    register_ants(fixed_img=path_to_img, moving_img=path_to_atlas, output_prefix=atlas_reg_rigid_prefix,
                  registration_type='Rigid', image_ext=image_ext, fixed_mask=fixed_mask)
    register_ants(fixed_img=path_to_img, moving_img=atlas_reg_rigid_prefix+image_ext,
                  output_prefix=atlas_reg_affine_prefix,
                  registration_type='Affine', image_ext=image_ext, fixed_mask=fixed_mask)
    #-- apply to brain_mask
    ants_apply_transforms(path_to_atlas_brain_mask, path_to_img, path_to_brainmask_reg_affine,
                          transforms=[affine_transform, rigid_transform])
    #-- skull strip
    ants_image_maths(path_to_img, path_to_brainmask_reg_affine, path_to_skullstripped, cmd='m')
    for img_path in also_apply_to:
        img_name_with_ext = os.path.basename(img_path)
        img_name, ext     = os.path.splitext(img_name_with_ext)
        path_to_skullstripped = os.path.join(output_dir, img_name + '_skullstripped' + image_ext)
        ants_image_maths(img_path, path_to_brainmask_reg_affine, path_to_skullstripped, cmd='m')



from datetime import datetime
import collections


def compile_registration_pairs_across_sessions(data_io, subject, ref_session, ref_seq='T1w'):
    print("  - compiling registration pairs across sessions for subject '%s', reference session '%s' reference sequence '%s'"%
                                                        (subject, ref_session, ref_seq))
    # -- reference image = fixed image
    fixed_img_files = data_io.get_image_files(subject=subject, session=ref_session, modality=ref_seq, processing='original')
    reg_tuple_list = []
    if len(fixed_img_files) == 1:
        fixed_img_file = fixed_img_files[0]
        # -- moving image
        unique_sessions_subject = data_io.bids_layout.get(target='session', subject=subject, return_type='id')
        print("session: %s"%unique_sessions_subject)
        for session in unique_sessions_subject:
            if not str(session) == str(ref_session):
                seq = ref_seq
                # -- try using same modality as ref_seq
                moving_img_file_list = data_io.get_image_files(subject=subject, session=session, modality=seq,
                                                               img_type='noreg')
                if not len(moving_img_file_list) > 0: # if session does not have re_seq
                    seq = data_io.get_reference_modality_for_session(subject, session,
                                                             ref_seq_list=['T1w-3D', 'T1w', 'T1wPost-3D', 'T1wPost', 'T2'])
                    moving_img_file_list = data_io.get_image_files(subject=subject, session=session, modality=seq,
                                            img_type = 'noreg')
                try:
                    moving_img_file = moving_img_file_list[0]

                    output_img_file = data_io.create_registered_image_path(registration='rigid', abs_path=True,
                                                                           subject=subject, session=session, modality=seq)
                    output_trafo_file = data_io.create_registration_transform_path(registration='rigid', abs_path=True,
                                                                                   subject=subject, session=session, modality=seq,
                                                                                   extension='mat')
                    reg_tuple = (fixed_img_file, moving_img_file, output_img_file, output_trafo_file)
                    print(moving_img_file, output_img_file, reg_tuple)
                    reg_tuple_list.append(reg_tuple)
                except:
                    print("could not define registration tuple for subject '%s', session '%s'"%(subject, session))
            else:
                output_img_file = data_io.create_registered_image_path(registration='reference', abs_path=True,
                                                                subject=subject, session=ref_session, modality=ref_seq,
                                                                extension=gt.get_file_extension(fixed_img_file))
                reg_tuple = (fixed_img_file, None, output_img_file, None)
                reg_tuple_list.append(reg_tuple)
    else:
        print("Multiple or no moving images found %s" % fixed_img_files)
    return reg_tuple_list


def compile_registration_pairs_within_session(data_io, seq_list, subject, session, ref_seq='T1w', use_reg_ref_seq=False,
                                              image_ext="nii"):
    print("  - compiling registration pairs within session '%s' for subject '%s', reference sequence '%s', use_reg_reg_seq='%s'"%
                                                        (session, subject, ref_seq,use_reg_ref_seq))
    # -- reference image = fixed image
    if use_reg_ref_seq:
        fixed_img_files = [data_io.create_registered_image_path(registration='rigid', abs_path=True,
                                                              subject=subject, session=session, modality=ref_seq,
                                                              extension=image_ext)]
    else:
        fixed_img_files = data_io.get_image_files(subject=subject, session=session, modality=ref_seq, processing='original')

    reg_tuple_list = []
    if len(fixed_img_files)==1:
        fixed_img_file = fixed_img_files[0]
        # -- moving image
        unique_sessions_subject = data_io.bids_layout.get(target='session', subject=subject, return_type='id')
        for seq in seq_list:
            if not seq == ref_seq:
                moving_img_file_list = data_io.get_image_files(subject=subject, session=session, modality=seq)
                if len(moving_img_file_list) > 0:
                    moving_img_file = moving_img_file_list[0]
                    output_img_file = data_io.create_registered_image_path(registration='rigid', abs_path=True,
                                                                       subject=subject, session=session, modality=seq)
                    output_trafo_file = data_io.create_registration_transform_path(registration='rigid', abs_path=True,
                                                                               subject=subject, session=session, modality=seq,
                                                                               extension='mat')
                    reg_tuple = (fixed_img_file, moving_img_file, output_img_file, output_trafo_file)
                    reg_tuple_list.append(reg_tuple)
            # only copy original image as reference if within-section reference has not been created by across-section registration
            elif seq==ref_seq and not use_reg_ref_seq:
                    output_img_file = data_io.create_registered_image_path(registration='reference', abs_path=True,
                                                                   subject=subject, session=session, modality=ref_seq,
                                                                   extension=gt.get_file_extension(fixed_img_file))
                    reg_tuple = (fixed_img_file, None, output_img_file, None)
                    reg_tuple_list.append(reg_tuple)
    else:
        print("Multiple or no moving images found %s"%fixed_img_files)

    return reg_tuple_list


def iterate_through_registration_tuples_and_register(reg_tuples, image_ext="nii", overwrite=False):
    for reg_tuple in reg_tuples:
        print("  - registation tuple: '( %s , %s , %s , %s )'"%(reg_tuple))
        fixed_img, moving_img, output, output_trafo_file = reg_tuple
        output_file = output + '.' + image_ext
        if (os.path.exists(output_file) and overwrite) or (not os.path.exists(output_file)):
            if moving_img is None: # moving_img=None -> reference image
                print("Copying reference image '%s' -> '%s'" % (fixed_img, output))
                shutil.copy(fixed_img, output)
            else:
                register_ants(fixed_img, moving_img, output, output_trafo_file, image_ext=image_ext)
        else:
            print("Output '%s' already exists ... skipping" % output_file)


def register_across_sessions(data_io, subject, ref_session, ref_seq='T1w', image_ext="nii", overwrite=False):
    print("== Registration across imaging sessions for subject '%s'"%subject)
    reg_tuples = compile_registration_pairs_across_sessions(data_io,
                                                            subject=subject, ref_session=ref_session, ref_seq=ref_seq)
    iterate_through_registration_tuples_and_register(reg_tuples, image_ext=image_ext, overwrite=overwrite)



def register_within_session(data_io, seq_list, subject, session, ref_seq='T1w', use_reg_ref_seq=False, image_ext=".nii",
                            overwrite=False):
    print("== Registration within imaging session '%s' for subject '%s'"%(session,subject))
    reg_tuples = compile_registration_pairs_within_session(data_io,
                                                           seq_list=seq_list, subject=subject, session=session,
                                                           ref_seq=ref_seq, use_reg_ref_seq=use_reg_ref_seq,
                                                           image_ext=image_ext)
    iterate_through_registration_tuples_and_register(reg_tuples, image_ext=image_ext, overwrite=overwrite)


def register_subject_anatomical(data_io, subject, image_ext="nii", overwrite=False, ref_session=None, ref_sequence=None,
                                use_ref_sequence_across_sessions=False,
                                seq_list = ['T2w', 'T2wFLAIR', 'T1wPost', 'T2w-3D', 'T1w-3D', 'T1wPost-3D', 'ADC']):
    unique_sessions_subject = data_io.bids_layout.get(target='session', subject=subject, return_type='id')
    if not ref_session:
        ref_session = data_io.get_reference_session_for_subject(subject)
    if not ref_sequence:
        ref_seq_across_sessions = data_io.get_reference_modality_for_session(subject, ref_session)
    else:
        ref_seq_across_sessions = ref_sequence
    print("== Register anatomical seuqences for subject '%s', reference session '%s', reference sequence '%s'"
                                                                %(subject, ref_session, ref_seq_across_sessions))
    #== register across sessions
    register_across_sessions(data_io, subject, ref_session, ref_seq=ref_seq_across_sessions,
                             image_ext=image_ext, overwrite=overwrite)
    #== register within sessions
    for session in unique_sessions_subject:
        if str(session)==str(ref_session):
            use_reg_ref_seq=False
            ref_seq = ref_seq_across_sessions
        else:
            use_reg_ref_seq = True
            if use_ref_sequence_across_sessions:
                ref_seq = ref_seq_across_sessions
            else:
                ref_seq = data_io.get_reference_modality_for_session(subject, session)
        register_within_session(data_io, seq_list, subject, session, ref_seq=ref_seq, use_reg_ref_seq=use_reg_ref_seq,
                                image_ext=image_ext,overwrite=overwrite)




def compile_registration_pairs_within_session_to_atlas(data_io, seq_list, subject, session, path_to_atlas,
                                              image_ext="nii"):
    print("  - compiling registration pairs within session '%s' for subject '%s', atlas '%s'"%
                                                        (session, subject, path_to_atlas))
    fixed_img_file = path_to_atlas
    # -- moving image
    reg_tuple_list = []
    unique_sessions_subject = data_io.bids_layout.get(target='session', subject=subject, return_type='id')
    for seq in seq_list:
        moving_img_file_list = data_io.get_image_files(subject=subject, session=session, modality=seq, processing='original')
        if len(moving_img_file_list) > 0:
            moving_img_file = moving_img_file_list[0]
            output_img_file = data_io.create_registered_image_path_to_atlas(registration='rigid', abs_path=True,
                                                                   subject=subject, session=session, modality=seq)
            output_trafo_file = data_io.create_registered_image_path_to_atlas(registration='rigid', abs_path=True,
                                                                           subject=subject, session=session, modality=seq,
                                                                           extension='mat')
            reg_tuple = (fixed_img_file, moving_img_file, output_img_file, output_trafo_file)
            reg_tuple_list.append(reg_tuple)
    return reg_tuple_list


def register_within_session_to_atlas(data_io, seq_list, subject, session, path_to_atlas, image_ext=".nii",
                            overwrite=False):
    print("== Registration within imaging session '%s' for subject '%s'"%(session,subject))
    reg_tuples = compile_registration_pairs_within_session_to_atlas(data_io,
                                                           seq_list=seq_list, subject=subject, session=session,
                                                           path_to_atlas=path_to_atlas,
                                                           image_ext=image_ext)
    iterate_through_registration_tuples_and_register(reg_tuples, image_ext=image_ext, overwrite=overwrite)


def register_subject_anatomical_to_atlas(data_io, subject, image_ext="nii", overwrite=False):
    seq_list = ['T2w', 'T2wFLAIR', 'T1wPost', 'T2w-3D', 'T1w-3D', 'T1wPost-3D', 'ADC']
    unique_sessions_subject = data_io.bids_layout.get(target='session', subject=subject, return_type='id')
    print("== Register anatomical sequences for subject '%s' to atlas"%(subject))
    path_to_atlas = config.path_to_atlas_t1_flipped
    #== register within sessions
    for session in unique_sessions_subject:
        register_within_session_to_atlas(data_io, seq_list, subject, session, path_to_atlas=path_to_atlas,
                                         image_ext=image_ext, overwrite=overwrite)



#=== SKULLSTRIPPING


def ants_skull_strip_image(path_to_image, path_to_mask, path_to_output):
    ants_image_maths(path_to_image, path_to_mask, path_to_output, cmd='m')


def ants_register_atlas_to_image(path_to_atlas, path_to_img,
                                 atlas_reg_rigid_prefix, atlas_reg_affine_prefix,
                                 path_to_rigid_transform, path_to_affine_transform,
                                 atlas_reg_deformable_prefix=None,
                                 path_to_deformable_transform=None,
                                 fixed_mask=None, image_ext='mha',
                                 apply_to_files = [],
                                 overwrite=False, verbose=0):
    print("== Starting ATLAS registrations ... ")
    # -- registrations
    print("  - Rigid registration ... ")
    if (os.path.exists(path_to_rigid_transform) and overwrite) or (not os.path.exists(path_to_rigid_transform)):
        register_ants(fixed_img=path_to_img, moving_img=path_to_atlas, output_prefix=atlas_reg_rigid_prefix,
                          path_to_transform=path_to_rigid_transform,fixed_mask=fixed_mask,
                          registration_type='Rigid', image_ext=image_ext, verbose=verbose)
    else:
        print("  - Rigid transformation '%s' already exists ... skipping" % (path_to_rigid_transform))

    print("  - Affine registration ... ")
    if (os.path.exists(path_to_affine_transform) and overwrite) or (not os.path.exists(path_to_affine_transform)):
        register_ants(fixed_img=path_to_img, moving_img=atlas_reg_rigid_prefix + '.' + image_ext,
                          output_prefix=atlas_reg_affine_prefix,
                          path_to_transform=path_to_affine_transform,
                          registration_type='Affine', image_ext=image_ext, fixed_mask=fixed_mask, verbose=verbose)
    else:
        print("  - Affine transformation '%s' already exists ... skipping" % (path_to_affine_transform))

    print("  - Deformable registration ... ")
    if atlas_reg_deformable_prefix and path_to_deformable_transform:
        if (os.path.exists(path_to_deformable_transform) and overwrite) or (not os.path.exists(path_to_deformable_transform)):
            register_ants(fixed_img=path_to_img, moving_img=atlas_reg_affine_prefix + '.' + image_ext,
                          output_prefix=atlas_reg_deformable_prefix,
                          path_to_transform=path_to_deformable_transform,
                          registration_type='Syn', image_ext=image_ext, fixed_mask=fixed_mask, verbose=verbose)
        else:
            print("  - Deformable transformation '%s' already exists ... skipping" % (path_to_deformable_transform))

    #== apply to other files
    out_path = os.path.dirname(atlas_reg_rigid_prefix)
    for file in apply_to_files:
        if os.path.exists(path_to_rigid_transform):
            output_file_rigid = os.path.join(out_path, os.path.basename(file).split('.')[0]+'_rigid'+'.'+image_ext)
            print("  - Applying rigid transform to '%s' -> '%s'"%(file, output_file_rigid))
            ants_apply_transforms(file, path_to_img, output_file_rigid, [path_to_rigid_transform])
            if os.path.exists(path_to_affine_transform):
                output_file_affine = os.path.join(out_path, os.path.basename(file).split('.')[0]+'_affine'+'.'+image_ext)
                print("  - Applying affine transform to '%s' -> '%s'"%(file, output_file_affine))
                ants_apply_transforms(file, path_to_img, output_file_affine,
                                      [path_to_affine_transform,path_to_rigid_transform])
                if os.path.exists(path_to_deformable_transform):
                    output_file_def = os.path.join(out_path, os.path.basename(file).split('.')[0]+'_def'+'.'+image_ext)
                    print("  -Applying deformable transform to '%s' -> '%s'"%(file, output_file_def))
                    ants_apply_transforms(file, path_to_img, output_file_def,
                                       [path_to_deformable_transform,path_to_affine_transform,path_to_rigid_transform])




def skullstrip_image(data_io, path_to_atlas, path_to_atlas_brainmask,
                     subject, session, modality='T1w', reg_type=None, overwrite=False, img_extension='nii', **kwargs):
    # == identify file
    if reg_type:
        files = data_io.get_image_files(subject=subject, session=session, modality=modality, registration=reg_type,
                                        **kwargs)
        path_to_img_skullstripped_rigid = data_io.create_registered_image_path(
            subject=subject, session=session, modality=modality,
            registration=reg_type, other='skullstripped-rigid',
            extension=img_extension)
        path_to_img_skullstripped_affine = data_io.create_registered_image_path(
            subject=subject, session=session, modality=modality,
            registration=reg_type, other='skullstripped-affine',
            extension=img_extension)
    else:
        files = data_io.get_image_files(subject=subject, session=session, modality=modality, img_type='noreg', **kwargs)
        path_to_img_skullstripped_rigid = data_io.create_image_path(
            subject=subject, session=session, modality=modality,
            other='skullstripped-rigid', extension=img_extension)
        path_to_img_skullstripped_affine = data_io.create_image_path(
            subject=subject, session=session, modality=modality,
            other='skullstripped-affine', extension=img_extension)

    if len(files) == 1:
        path_to_img = files[0]
    elif len(files) > 1:
        print("Multiple matching files: %s  -- refine your query" % (files))
    else:
        print("Did not find file... stopping.")
        return 0
    # == create various pathnames
    atlas_reg_rigid_prefix = data_io.create_registered_image_path(registration='rigid', alternative_name='atlas',
                                                                  subject=subject, session=session, modality=modality)
    atlas_reg_affine_prefix = data_io.create_registered_image_path(registration='affine', alternative_name='atlas',
                                                                   subject=subject, session=session, modality=modality)
    path_to_rigid_transform = data_io.create_registration_transform_path(
        registration='rigid', alternative_name='atlas',
        subject=subject, session=session, modality=modality,
        extension='mat')
    path_to_affine_transform = data_io.create_registration_transform_path(
        registration='affine', alternative_name='atlas',
        subject=subject, session=session, modality=modality,
        extension='mat')
    path_to_brainmask_reg_rigid = data_io.create_segmentation_image_path(
        registration='rigid', alternative_name='atlas',
        subject=subject, session=session, modality=modality,
        segmentation="brainmask", extension='mha')
    path_to_brainmask_reg_affine = data_io.create_segmentation_image_path(
        registration='affine', alternative_name='atlas',
        subject=subject, session=session, modality=modality,
        segmentation="brainmask", extension='mha')
    # == perform registration
    # == perform registration
    output_prefix = os.path.join(os.path.dirname(atlas_reg_affine_prefix), 'out')
    path_to_trafo_init = output_prefix + '0DerivedInitialMovingTranslation.mat'
    path_to_trafo_rigid  = output_prefix + '1Rigid.mat'
    path_to_trafo_affine = output_prefix + '2Affine.mat'
    path_to_trafo_warp      = output_prefix+'3Warp.nii.gz'
    if os.path.exists(path_to_trafo_init) and os.path.exists(path_to_trafo_rigid) \
            and os.path.exists(path_to_trafo_affine) and os.path.exists(path_to_trafo_warp):
        if overwrite:
            register_ants_synquick(path_to_img, path_to_atlas, output_prefix, registration='a')
    else:
        register_ants_synquick(path_to_img, path_to_atlas, output_prefix, registration='a')
    #== transform brainmask and skullstrip
    ants_apply_transforms(path_to_atlas_brainmask, path_to_img, path_to_brainmask_reg_affine,
                          transforms=[path_to_trafo_affine, path_to_trafo_rigid, path_to_trafo_init])
    ants_skull_strip_image(path_to_img, path_to_brainmask_reg_affine, path_to_img_skullstripped_affine)
    ants_apply_transforms(path_to_atlas_brainmask, path_to_img, path_to_brainmask_reg_rigid,
                          transforms=[path_to_trafo_rigid, path_to_trafo_init])
    ants_skull_strip_image(path_to_img, path_to_brainmask_reg_rigid, path_to_img_skullstripped_rigid)







def skullstrip_image_old(data_io, path_to_atlas, path_to_atlas_brainmask,
                     subject, session, modality='T1w', reg_type=None, overwrite=False, img_extension='nii', **kwargs):
    """
    This has been replaced by 'skullstrip_image' which uses synquick.
    After affine registration, that method handles eye removal better
    """
    # == identify file
    if reg_type:
        files = data_io.get_image_files(subject=subject, session=session, modality=modality, registration=reg_type,
                                        **kwargs)
        path_to_img_skullstripped_rigid = data_io.create_registered_image_path(
            subject=subject, session=session, modality=modality,
            registration=reg_type, other='skullstripped-rigid',
            extension=img_extension)
        path_to_img_skullstripped_affine = data_io.create_registered_image_path(
            subject=subject, session=session, modality=modality,
            registration=reg_type, other='skullstripped-affine',
            extension=img_extension)
    else:
        files = data_io.get_image_files(subject=subject, session=session, modality=modality, img_type='noreg', **kwargs)
        path_to_img_skullstripped_rigid = data_io.create_image_path(
            subject=subject, session=session, modality=modality,
            other='skullstripped-rigid', extension=img_extension)
        path_to_img_skullstripped_affine = data_io.create_image_path(
            subject=subject, session=session, modality=modality,
            other='skullstripped-affine', extension=img_extension)

    if len(files) == 1:
        path_to_img = files[0]
    elif len(files) > 1:
        print("Multiple matching files: %s  -- refine your query" % (files))
    else:
        print("Did not find file... stopping.")
        return 0
    # == create various pathnames
    atlas_reg_rigid_prefix = data_io.create_registered_image_path(registration='rigid', alternative_name='atlas',
                                                                  subject=subject, session=session, modality=modality)
    atlas_reg_affine_prefix = data_io.create_registered_image_path(registration='affine', alternative_name='atlas',
                                                                   subject=subject, session=session, modality=modality)
    path_to_rigid_transform = data_io.create_registration_transform_path(
        registration='rigid', alternative_name='atlas',
        subject=subject, session=session, modality=modality,
        extension='mat')
    path_to_affine_transform = data_io.create_registration_transform_path(
        registration='affine', alternative_name='atlas',
        subject=subject, session=session, modality=modality,
        extension='mat')
    path_to_brainmask_reg_rigid = data_io.create_segmentation_image_path(
        registration='rigid', alternative_name='atlas',
        subject=subject, session=session, modality=modality,
        segmentation="brainmask", extension='mha')
    path_to_brainmask_reg_affine = data_io.create_segmentation_image_path(
        registration='affine', alternative_name='atlas',
        subject=subject, session=session, modality=modality,
        segmentation="brainmask", extension='mha')
    # == perform registration
    ants_register_atlas_to_image(path_to_atlas, path_to_img,
                                 atlas_reg_rigid_prefix, atlas_reg_affine_prefix,
                                 path_to_rigid_transform, path_to_affine_transform,
                                 overwrite=overwrite)

    # == transform brain mask
    # rigid_transform_path = atlas_reg_rigid_prefix + '0GenericAffine.mat'
    # affine_transform = atlas_reg_affine_prefix + '0GenericAffine.mat'
    ants_apply_transforms(path_to_atlas_brainmask, path_to_img, path_to_brainmask_reg_rigid,
                              transforms=[path_to_rigid_transform])
    ants_apply_transforms(path_to_atlas_brainmask, path_to_img, path_to_brainmask_reg_affine,
                              transforms=[path_to_affine_transform, path_to_rigid_transform])
    # == skullstrip
    print(path_to_img)
    ants_skull_strip_image(path_to_img, path_to_brainmask_reg_rigid, path_to_img_skullstripped_rigid)
    ants_skull_strip_image(path_to_img, path_to_brainmask_reg_affine, path_to_img_skullstripped_affine)




def create_ventricle_references(data_io, path_to_atlas, path_to_atlas_ventricles,
                                subject, session, modality='T1w', reg_type=None, mask=None,
                                overwrite=False, img_extension='nii', other='withskull', output_other='withskull',
                                verbose=0, processing='registered', output_processing='registered',
                                output_modality='VentricleRegistration', **kwargs):
    print("== Creating reference & deformed ventricle segmentations ")
    # == identify reference file to which atlas is to be registered
    if reg_type:
        files = data_io.get_image_files(subject=subject, session=session, modality=modality, registration=reg_type,
                                        other=other, processing=processing, name='standard', **kwargs)
    else:
        files = data_io.get_image_files(subject=subject, session=session, modality=modality, img_type='noreg',
                                        other=other, name='standard', **kwargs)
    if len(files) == 1:
        path_to_img = files[0]
    elif len(files) > 1:
        print("Multiple matching files: %s  -- refine your query" % (files))
    else:
        print("Did not find file ... stopping.")
        return
    # == create various pathnames
    atlas_reg_rigid_prefix = data_io.create_registered_image_path(registration='rigid', alternative_name='atlas',
                                                                  subject=subject, session=session, modality=output_modality,
                                                                  other=output_other, processing=output_processing)
    atlas_reg_affine_prefix = data_io.create_registered_image_path(registration='affine', alternative_name='atlas',
                                                                   subject=subject, session=session, modality=output_modality,
                                                                   other=output_other, processing=output_processing)
    atlas_reg_def_prefix = data_io.create_registered_image_path(registration='deformable', alternative_name='atlas',
                                                                   subject=subject, session=session, modality=output_modality,
                                                                    other=output_other, processing=output_processing)
    path_to_rigid_transform = data_io.create_registration_transform_path(
        registration='rigid', alternative_name='atlas',
        subject=subject, session=session, modality=output_modality,
        extension='mat', processing=output_processing)
    path_to_affine_transform = data_io.create_registration_transform_path(
        registration='affine', alternative_name='atlas',
        subject=subject, session=session, modality=output_modality,
        extension='mat', processing=output_processing)
    path_to_def_transform = data_io.create_registration_transform_path(
        registration='deformable', alternative_name='atlas',
        subject=subject, session=session, modality=output_modality,
        extension='nii.gz', processing=output_processing)
    path_to_ventricles_reg_rigid = data_io.create_segmentation_image_path(
        registration='rigid', alternative_name='atlas',
        subject=subject, session=session, modality=output_modality,
        segmentation="ventricles", extension='mha', processing=output_processing)
    path_to_ventricles_reg_affine = data_io.create_segmentation_image_path(
        registration='affine', alternative_name='atlas',
        subject=subject, session=session, modality=output_modality,
        segmentation="ventricles", extension='mha', processing=output_processing)
    path_to_ventricles_reg_def = data_io.create_segmentation_image_path(
        registration='deformable', alternative_name='atlas',
        subject=subject, session=session, modality=output_modality,
        segmentation="ventricles", extension='mha', processing=output_processing)
    # == perform registration
    ants_register_atlas_to_image(path_to_atlas, path_to_img,
                                     atlas_reg_rigid_prefix, atlas_reg_affine_prefix,
                                     path_to_rigid_transform, path_to_affine_transform,
                                     atlas_reg_deformable_prefix=atlas_reg_def_prefix,
                                     path_to_deformable_transform=path_to_def_transform,
                                     fixed_mask=mask, image_ext='mha', overwrite=overwrite,
                                    verbose=verbose)

    # == transform ventricles
    ants_apply_transforms(path_to_atlas_ventricles, path_to_img, path_to_ventricles_reg_rigid,
                              transforms=[path_to_rigid_transform])
    ants_apply_transforms(path_to_atlas_ventricles, path_to_img, path_to_ventricles_reg_affine,
                              transforms=[path_to_affine_transform, path_to_rigid_transform])
    ants_apply_transforms(path_to_atlas_ventricles, path_to_img, path_to_ventricles_reg_def,
                              transforms=[path_to_def_transform, path_to_affine_transform, path_to_rigid_transform])




def create_ventricle_references_new(data_io, path_to_atlas, path_to_atlas_ventricles,
                                subject, session, modality='T1w', reg_type=None, mask=None,
                                overwrite=False, img_extension='nii', other='withskull', output_other='withskull',
                                verbose=0, processing='registered', output_processing='registered',
                                output_modality='VentricleRegistration', **kwargs):
    print("== Creating reference & deformed ventricle segmentations ")
    # == identify reference file to which atlas is to be registered
    if reg_type:
        files = data_io.get_image_files(subject=subject, session=session, modality=modality, registration=reg_type,
                                        other=other, processing=processing, name='standard', **kwargs)
    else:
        files = data_io.get_image_files(subject=subject, session=session, modality=modality, img_type='noreg',
                                        other=other, name='standard', **kwargs)
    print(files)
    if len(files) == 1:
        path_to_img = files[0]
    elif len(files) > 1:
        print("Multiple matching files: %s  -- refine your query" % (files))
    else:
        print("Did not find file... stopping.")
        return 0
    # == create various pathnames
    atlas_reg_affine = data_io.create_registered_image_path(registration='affine', alternative_name='atlas',
                                                                   subject=subject, session=session, modality=output_modality,
                                                                   other=output_other, processing=output_processing, extension=img_extension,
                                                                    **kwargs)
    atlas_reg_def = data_io.create_registered_image_path(registration='deformable', alternative_name='atlas',
                                                                   subject=subject, session=session, modality=output_modality,
                                                                    other=output_other, processing=output_processing, extension=img_extension,
                                                                    **kwargs)
    path_to_rigid_transform = data_io.create_registration_transform_path(
        registration='rigid', alternative_name='atlas',
        subject=subject, session=session, modality=output_modality,
        extension='mat', processing=output_processing, **kwargs)
    path_to_affine_transform = data_io.create_registration_transform_path(
        registration='affine', alternative_name='atlas',
        subject=subject, session=session, modality=output_modality,
        extension='mat', processing=output_processing, **kwargs)
    path_to_def_transform = data_io.create_registration_transform_path(
        registration='deformable', alternative_name='atlas',
        subject=subject, session=session, modality=output_modality,
        extension='nii.gz', processing=output_processing, **kwargs)
    path_to_ventricles_reg_rigid = data_io.create_segmentation_image_path(
        registration='rigid', alternative_name='atlas',
        subject=subject, session=session, modality=output_modality,
        segmentation="ventricles", extension='mha', processing=output_processing,
        **kwargs)
    path_to_ventricles_reg_affine = data_io.create_segmentation_image_path(
        registration='affine', alternative_name='atlas',
        subject=subject, session=session, modality=output_modality,
        segmentation="ventricles", extension='mha', processing=output_processing,
        **kwargs)
    path_to_ventricles_reg_def = data_io.create_segmentation_image_path(
        registration='deformable', alternative_name='atlas',
        subject=subject, session=session, modality=output_modality,
        segmentation="ventricles", extension='mha', processing=output_processing,
        **kwargs)
    # == perform registration
    output_prefix = os.path.join(os.path.dirname(path_to_affine_transform), 'out')
    path_to_trafo_init = output_prefix + '0DerivedInitialMovingTranslation.mat'
    path_to_trafo_rigid  = output_prefix + '1Rigid.mat'
    path_to_trafo_affine = output_prefix + '2Affine.mat'
    path_to_trafo_warp      = output_prefix+'3Warp.nii.gz'
    if os.path.exists(path_to_trafo_init) and os.path.exists(path_to_trafo_rigid) \
            and os.path.exists(path_to_trafo_affine) and os.path.exists(path_to_trafo_warp):
        if overwrite:
            register_ants_synquick(path_to_img, path_to_atlas, output_prefix, registration='s', fixed_mask=mask)
    else:
        register_ants_synquick(path_to_img, path_to_atlas, output_prefix, registration='s', fixed_mask=mask)

    # == transform Atlas
    ants_apply_transforms(path_to_atlas, path_to_img, atlas_reg_affine,
                          transforms=[path_to_trafo_affine, path_to_trafo_rigid, path_to_trafo_init])
    ants_apply_transforms(path_to_atlas, path_to_img, atlas_reg_def,
                          transforms=[path_to_trafo_warp, path_to_trafo_affine, path_to_trafo_rigid, path_to_trafo_init])

    # == transform ventricles
    ants_apply_transforms(path_to_atlas_ventricles, path_to_img, path_to_ventricles_reg_affine,
                              transforms=[path_to_trafo_affine, path_to_trafo_rigid, path_to_trafo_init])
    ants_apply_transforms(path_to_atlas_ventricles, path_to_img, path_to_ventricles_reg_def,
                              transforms=[path_to_trafo_warp, path_to_trafo_affine, path_to_trafo_rigid, path_to_trafo_init])






# skullstrip_image(subject, session, modality, reg_type='reference', other='withskull', overwrite=True)

def skullstrip_images_in_session(data_io, subject, session, path_to_atlas, path_to_atlas_brainmask,
                                 img_extension='nii', overwrite=False):
    print("== SKULLSTRIPPING IMAGES for patient '%s', session '%s'" % (subject, session))
    ref_seq = data_io.get_reference_modality_for_session(subject, session)
    print("   - Computing for reference file '%s'" % ref_seq)
    # -- Check if image for reference sequence is copied (registration='reference') or registered (registration='rigid')
    filelist = data_io.bids_layout.get(subject=subject, session=session, modality=ref_seq,
                                       other='withskull', registration='reference')
    if len(filelist) == 1:
        skullstrip_image(data_io, path_to_atlas, path_to_atlas_brainmask,
                         subject, session, ref_seq, reg_type='reference', other='withskull', overwrite=overwrite)
    else:
        skullstrip_image(data_io, path_to_atlas, path_to_atlas_brainmask,
                         subject, session, ref_seq, reg_type='rigid', other='withskull', overwrite=overwrite)
    # apply mask to other registered sequences in this session
    sequences = data_io.bids_layout.get(target='modality', subject=subject, registration='rigid',
                                        session=session, return_type='id')
    for sequence in sequences:
        if not sequence == ref_seq:
            print("   - Applying to sequence '%s'" % sequence)
            path_to_img = data_io.create_registered_image_path(
                subject=subject, session=session, modality=sequence,
                registration='rigid', extension=img_extension
            )
            path_to_brainmask_reg_rigid = data_io.create_segmentation_image_path(
                registration='rigid', alternative_name='atlas',
                subject=subject, session=session, modality=ref_seq,
                segmentation="brainmask", extension='mha')
            path_to_brainmask_reg_affine = data_io.create_segmentation_image_path(
                registration='affine', alternative_name='atlas',
                subject=subject, session=session, modality=ref_seq,
                segmentation="brainmask", extension='mha')
            path_to_img_skullstripped_rigid = data_io.create_registered_image_path(
                subject=subject, session=session, modality=sequence,
                registration='rigid', other='skullstripped-rigid',
                extension=img_extension)
            path_to_img_skullstripped_affine = data_io.create_registered_image_path(
                subject=subject, session=session, modality=sequence,
                registration='affine', other='skullstripped-affine',
                extension=img_extension)
            # print(path_to_img, path_to_brainmask_reg_rigid, path_to_img_skullstripped_rigid)
            ants_skull_strip_image(path_to_img, path_to_brainmask_reg_rigid, path_to_img_skullstripped_rigid)
            ants_skull_strip_image(path_to_img, path_to_brainmask_reg_affine, path_to_img_skullstripped_affine)


#===== OLD REGISTRATION APPORACHES -- CHECK FOR REGISTRATION OF FUNCTIONAL SEQUENCES

def register_sequences_across_sessions(ses_list, input_base_dir, subject, ref_ses, output_base_dir=None, ref_seq='T1w',
                                       image_ext='.mha', reg_name_suffix='_REG'):
    if not output_base_dir:
        output_base_dir = input_base_dir
    print("== Registration across imaging sessions for patient '%s'"%subject)
    subject_data_dir    = os.path.join(input_base_dir, subject)
    session_folders     = os.listdir(subject_data_dir)
    ref_img_name    = subject + "_" + ref_ses + "_" + ref_seq + '.nii'
    path_to_ref_img = os.path.join(subject_data_dir, ref_ses, ref_seq, ref_img_name)
    for session in ses_list:
        session_dir          = os.path.join(subject_data_dir, session)
        moving_image_name    = subject + "_" + session + "_" + ref_seq + '.nii'
        path_to_moving_image = os.path.join(session_dir, ref_seq, moving_image_name)
        if os.path.exists(path_to_moving_image):
            print("   -- Processing session '%s', sequence '%s'"%(session, ref_seq))
            reg_image_name  = subject + "_" + session + "_" + ref_seq + reg_name_suffix
            output_prefix = os.path.join(output_base_dir, subject, session, ref_seq,reg_image_name)
            register_ants(path_to_ref_img, path_to_moving_image, output_prefix, image_ext=image_ext)


def register_sequences_at_timepoint(seq_list, input_base_dir, subject, session, output_base_dir=None, ref_seq='T1w',
                                    use_reg_ref_seq=False, image_ext='.mha', reg_name_suffix='_REG'):
    if not output_base_dir:
        output_base_dir = input_base_dir
    print("== Registration within imaging session for patient '%s', session '%s'"%(subject, session))
    current_dir = os.path.join(input_base_dir, subject, session)
    if use_reg_ref_seq:
        fixed_image_name        = subject + "_" + session + "_" + ref_seq+reg_name_suffix+'.nii'
    else:
        fixed_image_name = subject + "_" + session + "_" + ref_seq + '.nii'
    path_to_fixed_img       = os.path.join(current_dir, ref_seq, fixed_image_name)
    if os.path.exists(path_to_fixed_img):
        for seq in seq_list:
            if not seq==ref_seq:
                moving_image_name       = subject + "_" + session + "_" + seq+'.nii'
                path_to_moving_img      = os.path.join(current_dir, seq, moving_image_name)
                if os.path.exists(path_to_moving_img):
                    print("   -- Processing sequence '%s'" % seq)
                    reg_image_name    = subject + "_" + session + "_" + seq + reg_name_suffix
                    output_prefix = os.path.join(output_base_dir, subject, session, seq, reg_image_name)
                    try:
                        register_ants(path_to_fixed_img, path_to_moving_img, output_prefix, image_ext=image_ext)
                    except:
                        e = sys.exc_info()[0]
                        print("Could not perform registration for patient '%s', session '%s', sequence '%s'"%
                                                            (subject, session, seq))
                        print("Error: %s"%e)

        else:
            print("-- Reference sequence '%s' does not exist for session '%s'"%(ref_seq, session))


def register_patient(input_base_dir, subject, output_base_dir=None, ref_seq='T1w',
                     session_prefix='session_', image_ext='.nii', reg_name_suffix='_REG'):
    if not output_base_dir:
        output_base_dir = input_base_dir
    print("%% Processing patient '%s'"%subject)
    patient_data_dir    = os.path.join(input_base_dir, subject)
    session_folders     = os.listdir(patient_data_dir)
    print(session_folders)
    # get dates corresponding to acquisition time points
    date_folder_map = collections.OrderedDict(
        {datetime.strptime(dir[len(session_prefix):], '%Y-%m-%d'): dir for dir in session_folders if not dir.startswith('.')})
    # use first date as reference TP
    reference_date = sorted(date_folder_map)[0]
    reference_session = date_folder_map[reference_date]
    ses_list = [ses for ses in session_folders if not ses == reference_session]
    # resave reference image
    if type(ref_seq) == str:
        ref_seq = [ref_seq]
    ref_seq_ = None
    for i in range(len(ref_seq)):
        ref_seq_tmp = ref_seq[i]
        print(ref_seq_tmp)
        path_to_ref_file_dir = os.path.join(patient_data_dir, reference_session, ref_seq_tmp)
        ref_base_file_name = subject + "_" + reference_session + "_" + ref_seq_tmp +".nii"
        print(os.path.join(path_to_ref_file_dir, ref_base_file_name))
        if os.path.exists(os.path.join(path_to_ref_file_dir, ref_base_file_name)):
            ref_seq_ = ref_seq_tmp
            break
    if not ref_seq_==None:
        ref_base_file_name_no_extension = ref_base_file_name.split('.')[0]
        print(ref_base_file_name_no_extension)
        path_in = os.path.join(path_to_ref_file_dir, ref_base_file_name_no_extension + '.nii')
        path_out= os.path.join(path_to_ref_file_dir, ref_base_file_name_no_extension + reg_name_suffix + image_ext)
        shutil.copy(path_in,path_out )
        #os.system("cp %s %s"%(path_in, path_out))
        # register_simple T1w across TPs
        register_sequences_across_sessions(ses_list, input_base_dir=input_base_dir, subject=subject,
                                           ref_ses=reference_session,
                                           output_base_dir=output_base_dir,
                                           ref_seq=ref_seq_, image_ext=image_ext,
                                           reg_name_suffix=reg_name_suffix)
        # register_simple sequences at each TP to T1w_REG of the respective TP
        seq_list = ['T2w', 'T2wFLAIR', 'T1wPost', 'T2w-3D', 'T1w-3D', 'T1wPost-3D', 'ADC']
        for session in session_folders:
            if not session.startswith('.'):
                if session == reference_session:
                    register_sequences_at_timepoint(seq_list, input_base_dir=input_base_dir, subject=subject,
                                                    session=session,
                                                    output_base_dir=output_base_dir,
                                                    ref_seq=ref_seq_, use_reg_ref_seq=False,
                                                    image_ext=image_ext,
                                                    reg_name_suffix=reg_name_suffix)
                else:
                    register_sequences_at_timepoint(seq_list, input_base_dir=input_base_dir, subject=subject,
                                                    session=session,
                                                    output_base_dir=output_base_dir,
                                                    ref_seq=ref_seq_, use_reg_ref_seq=True,
                                                    image_ext=image_ext,
                                                    reg_name_suffix=reg_name_suffix)

                # align DCE
                print("   -- DCE")
                ref_reg_file_name       = subject + "_" + reference_session + "_" + ref_seq_ + "_REF.nii"
                path_to_ref_reg_file    = os.path.join(input_base_dir, subject, reference_session, ref_seq_, ref_reg_file_name)
                dce_file_name           = subject + "_" + session + "_" + 'DCE' + ".nii"
                path_to_dce_file        = os.path.join(input_base_dir, subject, session, 'DCE', dce_file_name)
                output_path = os.path.join(output_base_dir, subject, session, 'DCE', 'DCE_REG')
                gt.ensure_dir_exists(output_path)
                save_file_prefix = subject + "_" + session + "_" + 'DCE'
                split_and_register_dce(path_to_ref_reg_file, path_to_dce_file, output_path, save_file_prefix)
                # register_simple DCE parameter maps
                print("   -- DCE Parameter Maps")
                path_to_T1_file         = path_to_ref_reg_file
                path_to_DCE_file        = path_to_dce_file

                for dce_param_file in ['lambda', 'Ktr', 'vp']:
                    file_name   = subject + '_' + session + '_' + dce_param_file
                    path_to_dir = os.path.join(input_base_dir, subject, session, 'DCE-params')
                    path_to_file        = os.path.join(path_to_dir, file_name)
                    output_path         = path_to_dir
                    if os.path.exists(path_to_file):
                        ref_img = sitk.ReadImage(path_to_T1_file)
                        dce_img = sitk.ReadImage(path_to_DCE_file)
                        register_dce_param_map(dce_img, ref_img, path_to_raw_file=path_to_file,
                                            output_path=output_path, name=file_name)
    else:
        print("None of the refernece sequence options available")


def copy_registered_files(base_dir, target_dir):
    gt.ensure_dir_exists(target_dir)
    for dirName, subdirList, fileList in os.walk(base_dir):
        for filename in fileList:
            if filename.endswith('REG.nii') or filename.endswith('REF.nii'):
                path_to_file = os.path.join(dirName, filename)
                shutil.copy(path_to_file, target_dir)




def extract_tp(image_4d, tp):
    dir_4D              = np.asarray(image_4d.GetDirection()).reshape(4, 4)
    dir_3D              = dir_4D[:3, :3]
    spacing_3D          = image_4d.GetSpacing()[:3]
    origin_3D           = image_4d.GetOrigin()[:3]
    image_4d_npa        = sitk.GetArrayFromImage(image_4d)
    image_3d_npa        = image_4d_npa[tp,:,:,:]
    image_3d            = sitk.GetImageFromArray(image_3d_npa)
    image_3d.SetDirection(dir_3D.flatten())
    image_3d.SetSpacing(spacing_3D)
    image_3d.SetOrigin(origin_3D)
    return image_3d



def split_and_register_dce(path_to_reference, path_to_dce, output_dir, save_file_prefix=''):
    if os.path.exists(path_to_dce) and os.path.exists(path_to_reference):
        ref_img = sitk.ReadImage(path_to_reference)
        dce_img = sitk.ReadImage(path_to_dce)
        dce_tp_0 = extract_tp(dce_img, 0)
        transform_dce_tp_0_to_T1w = register_simple(ref_img, dce_tp_0)

        for i in range(dce_img.GetSize()[-1]):
            # extract
            dce_tp = extract_tp(dce_img, i)
            if not save_file_prefix=='':
                dce_tp_file_name = save_file_prefix+'_%02d.nii'%i
            else:
                dce_tp_file_name = "dce_%02d.nii" % i
            sitk.WriteImage(dce_tp, os.path.join(output_dir, dce_tp_file_name))
            # register_simple time-point i to time-point 0
            # ==> this does not improve alignment in many cases, in contrary
            # dce_tp_file_name_reg = "dce_%02d_REG_to_TP0.nii" % i
            # transform = register_simple(dce_tp_0, dce_tp)
            # save_transform_and_image(transform, dce_tp_0, dce_tp, output_dir, dce_tp_file_name_reg)

            # apply transformation from TP0 -> T1w to every TP
            resample = sitk.ResampleImageFilter()
            resample.SetReferenceImage(ref_img)
            resample.SetInterpolator(sitk.sitkLinear)
            resample.SetTransform(transform_dce_tp_0_to_T1w)
            dce_tp_reg = resample.Execute(dce_tp)

            if not save_file_prefix=='':
                dce_tp_file_name_reg_to_T1w = save_file_prefix+'_%02d_REG.nii'%i
            else:
                dce_tp_file_name_reg_to_T1w = "dce_%02d_REG.nii" % i
            sitk.WriteImage(dce_tp_reg, os.path.join(output_dir, dce_tp_file_name_reg_to_T1w))
        else:
            print("... file missing")






def read_prativas_dce_parameter_raw_file(path_to_raw_file, shape):
    # load raw file !!! MAKE SURE TO ADAPT THIS FOR THE SPECIFIC FILE YOU WANT TO LOAD
    img_raw_nda     = np.fromfile(path_to_raw_file, dtype='float32').byteswap() # apparently byteorder has to be swapped
    img_comp_nda_3D = img_raw_nda.reshape(shape)[:, :, ::-1]                   # and left right be mirrored
    return img_comp_nda_3D



def register_raw_file(path_to_raw_file, raw_ref_image, transformation, trafo_ref_image, output_path, name):
    """
    Expects raw_ref_image to have same dimensions and same original position and orientation as the raw file.
    The transformation describes a registration of the raw_ref_image relative to the trafo_ref_image
    to which the raw file should be aligned.

    Prativa's parameter files have 2 slices on top and bottom removed,
    i.e. these slices have to be removed from the reference image
    """
    # read raw file
    img_comp_nda_3D = read_prativas_dce_parameter_raw_file(path_to_raw_file,
                                                           sitk.GetArrayFromImage(raw_ref_image).shape)
    # transform to image and add header
    img_comp = sitk.GetImageFromArray(img_comp_nda_3D)
    img_comp.SetDirection(raw_ref_image.GetDirection())
    img_comp.SetSpacing(raw_ref_image.GetSpacing())
    img_comp.SetOrigin(raw_ref_image.GetOrigin())
    sitk.WriteImage(img_comp, os.path.join(output_path, name + '.nii'))
    # apply transformation
    save_transform_and_image(transformation, trafo_ref_image, img_comp,
                             output_path, name + '_REG.nii')


def register_dce_param_map(dce_img, ref_img, path_to_raw_file, output_path, name):
    dce_tp_0 = extract_tp(dce_img, 0)
    transform1 = register_simple(ref_img, dce_tp_0)
    dce_tp_0_truncated = dce_tp_0[:, :, 2:14]
    transform2 = register_simple(ref_img, dce_tp_0_truncated)
    save_transform_and_image(transform2, ref_img, dce_tp_0_truncated,
                             output_path, 'dce_tp0_trunc_reg')

    register_raw_file(path_to_raw_file=path_to_raw_file, raw_ref_image=dce_tp_0_truncated,
                  transformation=transform2, trafo_ref_image=ref_img,
                  output_path=output_path, name=name)


