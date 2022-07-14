import axios from 'axios';

const baseURL = 'http://localhost:5000';

export default {
  async getVideoStatus(videoID) {
    try {
      const response = await axios.get(`${baseURL}/api/v1/videos/${videoID}`);
      return response.data;
    } catch {
      return false;
    }
  },

  async getVideoInsights(videoID) {
    try {
      const response = await axios.get(`${baseURL}/api/v1/videos/${videoID}/insights`);
      return response.data;
    } catch {
      return false;
    }
  },

  async uploadVideo(formData) {
    try {
      const response = await axios.post(`${baseURL}/api/v1/videos`, formData);
      return response.data;
    } catch {
      return false;
    }
  },

  async startProcessingVideo(videoID, videoInfo) {
    try {
      const response = await axios.put(`${baseURL}/api/v1/videos/${videoID}`, videoInfo);
      return response.data;
    } catch {
      return false;
    }
  },

  async getAllReactionVideosInformation() {
    try {
      const response = await axios.get(`${baseURL}/api/v1/videos/reactions`);
      return response.data;
    } catch {
      return false;
    }
  },

  getVideoURL(videoID, processed) {
    return `${baseURL}/api/v1/videos/${videoID}/download?processed=${processed}`;
  },

  async getVideoKeymoments(videoID) {
    try {
      const response = await axios.get(`${baseURL}/api/v1/videos/${videoID}/key_moments`);
      return response.data;
    } catch {
      return false;
    }
  },

  async getVideoAnalytics(videoID) {
    try {
      const response = await axios.get(`${baseURL}/api/v1/videos/${videoID}/analytics`);
      return response.data;
    } catch {
      return false;
    }
  },
};
