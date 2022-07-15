<template>
  <zingchart :data="chartConfig" :series="data" />
</template>

<script>
import zingchartVue from 'zingchart-vue';

export default {
  props: {
    data: Array,
  },
  data() {
    return {
      chartConfig: {
        layout: 'horizontal',

        graphset: [
          {
            type: 'area',
            theme: 'dark',

            backgroundColor: 'none',
            plotarea: {
              'adjust-layout': true /* For automatic margin adjustment. */,
              backgroundColor: 'transparent',
            },

            scaleX: {
              // set scale label

              label: {
                text: 'Frames',
                fontSize: 16,
              },
            },
            scaleY: {
              // set scale label

              label: {
                text: 'Mood',
                fontSize: 16,
              },
            },

            plot: {
              marker: {
                visible: false,
              },
            },

            series: this.data,
          },
        ],
      },
    };
  },
  components: {
    zingchart: zingchartVue,
  },
};
</script>

<style></style>
