<template>
  <zingchart :data="chartConfig" :series="data" />
</template>

<script>
import zingchartVue from 'zingchart-vue';

export default {
  props: {
    data: Array,
    scaleY: Object,
  },
  data() {
    return {
      chartConfig: {
        layout: 'horizontal',

        graphset: [
          {
            type: 'line',
            theme: 'dark',

            backgroundColor: 'none',
            plotarea: {
              'adjust-layout': true /* For automatic margin adjustment. */,
              backgroundColor: 'transparent',
            },

            legend: {
              layout: 'float',
              'background-color': 'none',
              'border-width': 0,
              shadow: 0,
              align: 'center',
              'adjust-layout': true,
              'toggle-action': 'remove',
              item: {
                padding: 7,
                marginRight: 17,
                cursor: 'hand',
              },
            },

            plot: {
              marker: {
                visible: false,
              },
            },

            scaleX: {
              // set scale label

              label: {
                text: 'Seconds',
                fontSize: 16,
              },
            },
            scaleY: this.scaleY,

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
