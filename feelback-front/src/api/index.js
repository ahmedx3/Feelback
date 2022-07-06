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
};
