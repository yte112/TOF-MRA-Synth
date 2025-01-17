import os
import copy
import pydicom
import numpy as np


def save_series_with_template(pixel_data, templates, out_dir,
                              series_number_offset,
                              series_desc_suffix, reverse_hu=True, scale=1):
    """save pixel data to DICOM according to a template"""
    # print (pixel_data.shape, len(templates))
    assert len(pixel_data.shape) == 3 and pixel_data.shape[0] == len(templates)
    series_uid = pydicom.uid.generate_uid()
    uid_pool = set()
    uid_pool.add(series_uid)
    for i, data_set in enumerate(templates):
        uid_pool, out = save_slice_with_template(pixel_data[i], data_set, out_dir, i,
                                            series_number_offset,
                                            series_desc_suffix,
                                            series_uid, uid_pool, reverse_hu=reverse_hu, scale=scale)


def save_slice_with_template(pixel_data, template_dataset, out_dir, i_slice,
                             series_number_offset, series_desc_suffix,
                             series_uid, uid_pool, reverse_hu=True, scale=1):
    assert len(pixel_data.shape) == 2
    out_data_set = copy.deepcopy(template_dataset)
    data_shape = pixel_data.shape
    target_shape = template_dataset.pixel_array.shape
    if data_shape != target_shape:
        start = (target_shape[0] - data_shape[0]) // 2
        pixel_data = pixel_data[start:target_shape[0]+start, start:target_shape[0]+start]
    #print('out_data_set.PixelData_53_92',np.array(int(float(out_data_set.PixelData)),'int16')[92,53])
    sop_uid = pydicom.uid.generate_uid()
    while sop_uid in uid_pool:
        sop_uid = pydicom.uid.generate_uid()
    uid_pool.add(sop_uid)
    data_type = template_dataset.pixel_array.dtype
    # resolve the bits storation issue
    bits_stored = template_dataset.get("BitsStored", 16)
    if template_dataset.get("PixelRepresentation", 0) != 0:
        # signed
        t_min, t_max = (-(1 << (bits_stored - 1)), (1 << (bits_stored - 1)) - 1)
    else:
        # unsigned
        t_min, t_max = 0, (1 << bits_stored) - 1
    #print('t_min,t_max',t_min,t_max)

    # pixel_data dtype = float32!!!
    # RIntercept = out_data_set.RescaleIntercept
    # RSlope = out_data_set.RescaleSlope
    RIntercept = np.float64(out_data_set.RescaleIntercept)
    RSlope = np.float64(out_data_set.RescaleSlope)
    # print(RIntercept, RSlope)
    if reverse_hu:
        pixel_data =np.asarray(pixel_data - RIntercept) / RSlope

    pixel_data = np.array(pixel_data, dtype=np.int64)
    
    pixel_data[pixel_data < t_min] = t_min
    pixel_data[pixel_data > t_max] = t_max

    out_data_set.PixelData = pixel_data.astype(data_type).tostring()
    out_data_set.SeriesInstanceUID = series_uid
    out_data_set.SOPInstanceUID = sop_uid
    out_data_set.SeriesNumber = int(float(out_data_set.SeriesNumber)) + series_number_offset
    out_data_set.Rows = int(out_data_set.Rows * scale)
    out_data_set.Columns = int(out_data_set.Columns * scale)
    out_data_set.PixelSpacing = [out_data_set.PixelSpacing[0]/scale, out_data_set.PixelSpacing[1]/scale]
    # out_data_set.is_undefined_length = False
    # print(out_data_set.Rows, out_data_set.Columns, out_data_set.PixelSpacing[0], pixel_data.shape)
    if out_data_set.file_meta.TransferSyntaxUID.is_compressed:
        out_data_set.file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'
    if hasattr(out_data_set, 'SeriesDescription'):
        out_data_set.SeriesDescription += series_desc_suffix
    else:
        out_data_set.add_new([0x0008, 0x103e], 'LO', series_desc_suffix)
    out_path = os.path.join(out_dir, 'IMG_{:03d}.dcm'.format(i_slice))
    out_data_set.save_as(out_path)
    return uid_pool, out_data_set