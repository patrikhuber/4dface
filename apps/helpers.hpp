/*
 * 4dface: Real-time 3D face tracking and reconstruction from 2D video.
 *
 * File: apps/helpers.hpp
 *
 * Copyright 2015, 2016 Patrik Huber
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#ifndef APP_HELPERS_HPP_
#define APP_HELPERS_HPP_

#include "eos/core/Landmark.hpp"
#include "eos/core/LandmarkMapper.hpp"
#include "eos/morphablemodel/MorphableModel.hpp"
#include "eos/morphablemodel/Blendshape.hpp"
#include "eos/render/detail/render_detail.hpp"
#include "rcr/landmark.hpp"

#include "glm/mat4x4.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include "opencv2/core/core.hpp"

#include <vector>
#include <algorithm>
#include <iterator>
#include <cassert>

/**
 * @brief Scales and translates a facebox. Useful for converting
 * between face boxes from different face detectors.
 *
 * To convert from V&J faceboxes to ibug faceboxes, use a scaling
 * of 0.85 and a translation_y of 0.2.
 * Ideally, we would learn the exact parameters from data.
 *
 * @param[in] facebox Input facebox.
 * @param[in] scaling The input facebox will be scaled by this factor.
 * @param[in] translation_y How much, in percent of the original facebox's width, the facebox will be translated in y direction. A positive value means facebox moves downwards.
 * @return The rescaled facebox.
 */
cv::Rect rescale_facebox(cv::Rect facebox, float scaling, float translation_y)
{
	// Assumes a square input facebox to work? (width==height)
	const auto new_width = facebox.width * scaling;
	const auto smaller_in_px = facebox.width - new_width;
	const auto new_tl = facebox.tl() + cv::Point2i(smaller_in_px / 2.0f, smaller_in_px / 2.0f);
	const auto new_br = facebox.br() - cv::Point2i(smaller_in_px / 2.0f, smaller_in_px / 2.0f);
	cv::Rect rescaled_facebox(new_tl, new_br);
	rescaled_facebox.y += facebox.width * translation_y;
	return rescaled_facebox;
};

/**
 * @brief Converts an rcr::LandmarkCollection to an eos::core::LandmarkCollection.
 *
 * They are identical types, it would be great to remove that conversion, but then
 * we'd have to make one of the libraries dependent on the other, which I don't like.
 *
 * @param[in] landmark_collection Input rcr::LandmarkCollection.
 * @return Identical eos::core::LandmarkCollection.
 */
auto rcr_to_eos_landmark_collection(const rcr::LandmarkCollection<cv::Vec2f>& landmark_collection)
{
	eos::core::LandmarkCollection<cv::Vec2f> eos_landmark_collection;
	std::transform(begin(landmark_collection), end(landmark_collection), std::back_inserter(eos_landmark_collection), [](auto&& lm) { return eos::core::Landmark<cv::Vec2f>{ lm.name, lm.coordinates }; });
	return eos_landmark_collection;
};

/**
 * @brief Calculates the bounding box that encloses the landmarks.
 *
 * The bounding box will not be square.
 *
 * @param[in] landmarks Landmarks.
 * @return The enclosing bounding box.
 */
template<class T = int>
cv::Rect_<T> get_enclosing_bbox(cv::Mat landmarks)
{
	auto num_landmarks = landmarks.cols / 2;
	double min_x_val, max_x_val, min_y_val, max_y_val;
	cv::minMaxLoc(landmarks.colRange(0, num_landmarks), &min_x_val, &max_x_val);
	cv::minMaxLoc(landmarks.colRange(num_landmarks, landmarks.cols), &min_y_val, &max_y_val);
	return cv::Rect_<T>(min_x_val, min_y_val, max_x_val - min_x_val, max_y_val - min_y_val);
};

/**
 * @brief Makes the given face bounding box square by enlarging the
 * smaller of the width or height to be equal to the bigger one.
 *
 * @param[in] bounding_box Input bounding box.
 * @return The bounding box with equal width and height.
 */
cv::Rect make_bbox_square(cv::Rect bounding_box)
{
	auto center_x = bounding_box.x + bounding_box.width / 2.0;
	auto center_y = bounding_box.y + bounding_box.height / 2.0;
	auto box_size = std::max(bounding_box.width, bounding_box.height);
	return cv::Rect(center_x - box_size / 2.0, center_y - box_size / 2.0, box_size, box_size);
};

/**
 * Draws the given mesh as wireframe into the image.
 *
 * It does backface culling, i.e. draws only vertices in CCW order.
 *
 * @param[in] image An image to draw into.
 * @param[in] mesh The mesh to draw.
 * @param[in] modelview Model-view matrix to draw the mesh.
 * @param[in] projection Projection matrix to draw the mesh.
 * @param[in] viewport Viewport to draw the mesh.
 * @param[in] colour Colour of the mesh to be drawn.
 */
void draw_wireframe(cv::Mat image, const eos::render::Mesh& mesh, glm::mat4x4 modelview, glm::mat4x4 projection, glm::vec4 viewport, cv::Scalar colour = cv::Scalar(0, 255, 0, 255))
{
	for (const auto& triangle : mesh.tvi)
	{
		const auto p1 = glm::project({ mesh.vertices[triangle[0]][0], mesh.vertices[triangle[0]][1], mesh.vertices[triangle[0]][2] }, modelview, projection, viewport);
		const auto p2 = glm::project({ mesh.vertices[triangle[1]][0], mesh.vertices[triangle[1]][1], mesh.vertices[triangle[1]][2] }, modelview, projection, viewport);
		const auto p3 = glm::project({ mesh.vertices[triangle[2]][0], mesh.vertices[triangle[2]][1], mesh.vertices[triangle[2]][2] }, modelview, projection, viewport);
		if (eos::render::detail::are_vertices_ccw_in_screen_space(glm::vec2(p1), glm::vec2(p2), glm::vec2(p3)))
		{
			cv::line(image, cv::Point(p1.x, p1.y), cv::Point(p2.x, p2.y), colour);
			cv::line(image, cv::Point(p2.x, p2.y), cv::Point(p3.x, p3.y), colour);
			cv::line(image, cv::Point(p3.x, p3.y), cv::Point(p1.x, p1.y), colour);
		}
	}
};

/**
 * @brief Merges isomaps from a live video with a weighted averaging, based
 * on the view angle of each vertex to the camera.
 *
 * An optional merge_threshold can be specified upon construction. Pixels with
 * a view-angle above that threshold will be completely discarded. All pixels
 * below the threshold are merged with a weighting based on its vertex view-angle.
 * Assumes the isomaps to be 512x512.
 */
class WeightedIsomapAveraging
{
public:
	/**
	 * @brief Constructs a new object that will hold the current averaged isomap and
	 * be able to add frames from a live video and merge them on-the-fly.
	 *
	 * The threshold means: Each triangle with a view angle smaller than the given angle will be used to merge.
	 * The default threshold (90°) means all triangles, as long as they're a little bit visible, are merged.
	 *
	 * @param[in] merge_threshold View-angle merge threshold, in degrees, from 0 to 90.
	 */
	WeightedIsomapAveraging(float merge_threshold = 90.0f)
	{
		assert(merge_threshold >= 0.f && merge_threshold <= 90.f);

		visibility_counter = cv::Mat::zeros(512, 512, CV_32SC1);
		merged_isomap = cv::Mat::zeros(512, 512, CV_32FC4);

		// map 0° to 255, 90° to 0:
		float alpha_thresh = (-255.f / 90.f) * merge_threshold + 255.f;
		if (alpha_thresh < 0.f) // could maybe happen due to float inaccuracies / rounding?
			alpha_thresh = 0.0f;
		threshold = static_cast<unsigned char>(alpha_thresh);
	};

	/**
	 * @brief Merges the given new isomap with all previously processed isomaps.
	 *
	 * @param[in] isomap The new isomap to add.
	 * @return The merged isomap of all images processed so far, as 8UC4.
	 */
	cv::Mat add_and_merge(const cv::Mat& isomap)
	{
		// Merge isomaps, add the current to the already merged, pixel by pixel:
		for (int r = 0; r < isomap.rows; ++r)
		{
			for (int c = 0; c < isomap.cols; ++c)
			{
				if (isomap.at<cv::Vec4b>(r, c)[3] <= threshold)
				{
					continue; // ignore this pixel, not visible in the extracted isomap of this current frame
				}
				// we're sure to have a visible pixel, merge it:
				// merged_pixel = (old_average * visible_count + new_pixel) / (visible_count + 1)
				merged_isomap.at<cv::Vec4f>(r, c)[0] = (merged_isomap.at<cv::Vec4f>(r, c)[0] * visibility_counter.at<int>(r, c) + isomap.at<cv::Vec4b>(r, c)[0]) / (visibility_counter.at<int>(r, c) + 1);
				merged_isomap.at<cv::Vec4f>(r, c)[1] = (merged_isomap.at<cv::Vec4f>(r, c)[1] * visibility_counter.at<int>(r, c) + isomap.at<cv::Vec4b>(r, c)[1]) / (visibility_counter.at<int>(r, c) + 1);
				merged_isomap.at<cv::Vec4f>(r, c)[2] = (merged_isomap.at<cv::Vec4f>(r, c)[2] * visibility_counter.at<int>(r, c) + isomap.at<cv::Vec4b>(r, c)[2]) / (visibility_counter.at<int>(r, c) + 1);
				merged_isomap.at<cv::Vec4f>(r, c)[3] = 255; // as soon as we've seen the pixel visible once, we set it to visible.
				++visibility_counter.at<int>(r, c);
			}
		}
		cv::Mat merged_isomap_uchar;
		merged_isomap.convertTo(merged_isomap_uchar, CV_8UC4);
		return merged_isomap_uchar;
	};

private:
	cv::Mat visibility_counter;
	cv::Mat merged_isomap;
	unsigned char threshold;
};

/**
 * @brief Merges PCA coefficients from a live video with a simple averaging.
 */
class PcaCoefficientMerging
{
public:
	/**
	 * @brief Merges the given new PCA coefficients with all previously processed coefficients.
	 *
	 * @param[in] coefficients The new coefficients to add.
	 * @return The merged coefficients of all images processed so far.
	 */
	std::vector<float> add_and_merge(const std::vector<float>& coefficients)
	{
		if (merged_shape_coefficients.empty())
		{
			merged_shape_coefficients = cv::Mat::zeros(coefficients.size(), 1, CV_32FC1);
		}
		assert(coefficients.size() == merged_shape_coefficients.rows);

		cv::Mat test(coefficients);
		merged_shape_coefficients = (merged_shape_coefficients * num_processed_frames + test) / (num_processed_frames + 1.0f);
		++num_processed_frames;
		return std::vector<float>(merged_shape_coefficients.begin<float>(), merged_shape_coefficients.end<float>());
	};

private:
	int num_processed_frames = 0;
	cv::Mat merged_shape_coefficients;
};

#endif /* APP_HELPERS_HPP_ */
