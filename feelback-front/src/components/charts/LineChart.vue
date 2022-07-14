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
                text: 'Seconds',
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

            // series: this.data,
            series: [
              {
                aspect: 'spline',
                values: [
                  5, 6, 8, 10, 11, 12, 15, 20, 16, 14, 10, 5, 0, -3, -10, -15, -20,
                  -15, -13, -10, 0, 1, 3, 6, 8, 15, 20, 30, 21, 18, 59, 50, 28, 33,
                ],
              },
            ],
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
