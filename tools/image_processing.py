import SimpleITK as sitk
import numpy as np
import copy
import tools.general_tools as gt

def flip_image(image_in, flip_axis='x'):
    if flip_axis=='x':
        flip = [True, False, False]
    elif flip_axis=='y':
        flip = [False, True, False]
    elif flip_axis == 'z':
        flip = [False, False, True]
    img_flipped = sitk.Flip(image_in, flip)
    img_flipped.SetDirection(image_in.GetDirection())
    return img_flipped


def merge_segmentation_labels(input_img, label_map={}):
    """
    label_map: {<target value 1>: [<val to change 1>, <val to change 2>, ...],
                <target value 2>: [...]}
    """
    img_np = sitk.GetArrayFromImage(input_img)
    out_img_np = copy.deepcopy(img_np)
    for target_label, source_label_list in label_map.items():
        out_img_np[np.isin(img_np, source_label_list)]=target_label # NOTE: we check label values in input image img_np,
                                                                    #       not output image out_img_np!
    out_img = sitk.GetImageFromArray(out_img_np)
    out_img.SetSpacing(input_img.GetSpacing())
    out_img.SetDirection(input_img.GetDirection())
    out_img.SetOrigin(input_img.GetOrigin())
    return out_img


#== MANUAL METHOD, see sitk.LabelShapeStatisticsImageFilter() below
# def compute_volume(segmentation, label_tissue_map=None):
#     # tested against itk-snap label statistics -- same results
#     #segmentation = sitk.ReadImage(path_to_label_file)
#     spacing      = segmentation.GetSpacing()
#     vol_per_vox  = spacing[0]*spacing[1]*spacing[2]
#
#     segmentation_np = sitk.GetArrayFromImage(segmentation)
#     unique_labels   = np.unique(segmentation_np)
#     label_dict_count = {}
#     label_dict_volume= {}
#     for label in unique_labels:
#         label_img = copy.deepcopy(segmentation_np)
#         #label_img[np.where(label_img==label)]  = 1
#         label_img[np.where(label_img!=label)] = 0
#         label_cnt = np.count_nonzero(label_img)
#         label_dict_count[label] = label_cnt
#         label_dict_volume[label]= label_cnt*vol_per_vox
#
#     label_dict = {"count_label":label_dict_count,
#                   "volume_label":label_dict_volume,
#                   "volume_per_voxel":vol_per_vox}
#
#     if label_tissue_map:
#         label_dict["count_tissue"] = gt.map_between_dicts(label_dict['count_label'], label_tissue_map)
#         label_dict["volume_tissue"] = gt.map_between_dicts(label_dict['volume_label'], label_tissue_map)
#
#     return label_dict


def compute_volume(label_image, label_tissue_map=None, combine_labels=None):
    if combine_labels:
        label_image = merge_segmentation_labels(label_image, combine_labels)
    # tested against itk-snap label statistics -- same results
    spacing      = label_image.GetSpacing()
    vol_per_vox  = spacing[0]*spacing[1]*spacing[2]
    label_shape_stats = sitk.LabelShapeStatisticsImageFilter()
    label_shape_stats.Execute(label_image)
    label_dict_count = {}
    label_dict_volume= {}
    for label in label_shape_stats.GetLabels():
        label_volume = label_shape_stats.GetPhysicalSize(label)
        label_dict_count[label] = int(label_volume / vol_per_vox)
        label_dict_volume[label]= label_volume

    label_dict = {"count_label":label_dict_count,
                  "volume_label":label_dict_volume,
                  "volume_per_voxel":vol_per_vox}

    if label_tissue_map:
        label_dict["count_tissue"] = gt.map_between_dicts(label_dict['count_label'], label_tissue_map)
        label_dict["volume_tissue"] = gt.map_between_dicts(label_dict['volume_label'], label_tissue_map)

    return label_dict



def compute_center_of_mass(segmentation_img, label_id, combine_labels=None):
    if combine_labels:
        segmentation_img = merge_segmentation_labels(segmentation_img, combine_labels)
    #-- For standard image Orientation, the following 2 approaches are identical
    #-- 1) manual
    # import numpy as np
    # segmentation_img_np = sitk.GetArrayFromImage(segmentation_img)
    # index = np.where(segmentation_img_np==label_id)
    # z_img_center = index[0].sum()/index[0].size
    # y_img_center = index[1].sum()/index[1].size
    # x_img_center = index[2].sum()/index[2].size
    #
    # x_spacing, y_spacing, z_spacing = segmentation_img.GetSpacing()
    # x_origin, y_origin, z_origin = segmentation_img.GetOrigin()
    #
    # x_center = x_origin + x_img_center * x_spacing
    # y_center = y_origin + y_img_center * y_spacing
    # z_center = z_origin + z_img_center * z_spacing

    #-- 2) simpleITK.LabelShapeStatisticsImageFilter.GetCentroid
    label_shape_stats = sitk.LabelShapeStatisticsImageFilter()
    label_shape_stats.Execute(segmentation_img)
    x_center, y_center, z_center = label_shape_stats.GetCentroid(label_id)
    return x_center, y_center, z_center



def compute_lvd(path_to_ventricles_ref, path_to_ventricles_def, label_id=1, verbose=True):
    ventricles_ref = sitk.Cast(sitk.ReadImage(path_to_ventricles_ref), sitk.sitkUInt8)
    ventricles_def = sitk.Cast(sitk.ReadImage(path_to_ventricles_def), sitk.sitkUInt8)

    com_ref = compute_center_of_mass(ventricles_ref, label_id)
    com_def = compute_center_of_mass(ventricles_def, label_id)
    shift = np.array(com_ref) - np.array(com_def)
    lvd     = np.linalg.norm(shift)

    if verbose:
        print("==LVd computation for '%s'"%(path_to_ventricles_ref))
        print("                  ->  '%s'"%(path_to_ventricles_def))
        print("   - reference com: (%.2f, %.2f, %.2f)"%(com_ref))
        print("   - deformed com : (%.2f, %.2f, %.2f)" % (com_def))
        print("   - shift        : (%.2f, %.2f, %.2f)" % (shift[0], shift[1], shift[2]))
        print("   - LVd          :  %.2f [mm]" % (lvd))
    return lvd


def intersect_labels(seg_img_1, seg_img_2, label_1, label_2, mode='intersection', return_label=1):
    seg_img_1_np = sitk.GetArrayFromImage(seg_img_1)
    seg_img_2_np = sitk.GetArrayFromImage(seg_img_2)
    sel_1 = seg_img_1_np == label_1
    sel_2 = seg_img_2_np == label_2
    if mode == 'intersection':
        mask = np.logical_and(sel_1, sel_2)
    elif mode == 'union':
        mask = np.logical_or(sel_1, sel_2)
    elif mode == '1not2':
        mask = np.logical_and(sel_1, ~sel_2)

    out_label_img_np = np.zeros(seg_img_1_np.shape)
    out_label_img_np[mask] = return_label
    out_label_img = sitk.GetImageFromArray(out_label_img_np)

    out_label_img.SetSpacing(seg_img_1.GetSpacing())
    out_label_img.SetDirection(seg_img_1.GetDirection())
    out_label_img.SetOrigin(seg_img_1.GetOrigin())
    out_label_img = sitk.Cast(out_label_img, sitk.sitkUInt8)
    return out_label_img




def remove_tumor_from_ventricle_seg(ventricle_seg, tumor_seg, label_1=1, label_2=1, return_label=1):
    ventricles_wo_tumor = intersect_labels(ventricle_seg, tumor_seg, label_1=label_1, label_2=label_2,
                                              return_label=return_label, mode='1not2')
    #-- the resulting ventricle segmentation may have diconnected islands if the tumor volume does not
    #   entirely cover on part of the ventricular system.
    #   -> add filter to remove islands... possibly region growing with template base segmentation as reference
    return ventricles_wo_tumor