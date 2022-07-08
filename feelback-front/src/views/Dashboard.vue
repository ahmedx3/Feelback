<template>
  <Loading v-if="loading" :percentage="loadingPercentage" />
  <Analytics v-else :reactionVideoID="reactionVideoID" :trailerVideoID="trailerVideoID" />
</template>

<script>
import Analytics from '../components/Analytics.vue';
import Loading from '../components/Loading.vue';
import api from '../api/index';

export default {
  components: {
    Analytics,
    Loading,
  },

  data: () => ({
    loading: true,
    loadingPercentage: 0,

    // Video IDs
    reactionVideoID: null,
    trailerVideoID: null,
  }),

  methods: {
    async getVideoPercentage(videoID) {
      const response = await api.getVideoStatus(videoID);

      if (response && response.data.finished_processing) {
        // Set videos
        this.reactionVideoID = videoID;
        this.trailerVideoID = response.data.trailer_id;

        this.loadingPercentage = 100;
        this.loading = false;
      } else {
        this.loadingPercentage = response && response.data.progress ? response.data.progress : 0;
        setTimeout(() => {
          this.getVideoPercentage(videoID);
        }, 10000);
      }
    },
  },

  mounted() {
    // Get video ID from the url
    const videoID = this.$route.params.id;

    // Loop and send request to check that the video is done processing
    this.getVideoPercentage(videoID);
  },
};
</script>

<style></style>
