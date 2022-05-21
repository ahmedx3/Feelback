<template>
  <zingchart :data="chartConfig" />
</template>

<script>
import zingchartVue from 'zingchart-vue';

export default {
  data() {
    return {
      chartConfig: {
        layout: 'horizontal',

        graphset: [
          {
            type: 'line',
            theme: 'dark',

            plotarea: {
              'adjust-layout': true /* For automatic margin adjustment. */,
            },

            title: {
              text: 'Analytics',
              'font-size': '24px',
              'adjust-layout': true,
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

            scaleX: {
              // set scale label

              label: {
                text: 'Frame Number',

                fontSize: 16,
              },
            },
            scaleY: {
              // set scale label

              label: {
                text: 'Emotions',
                fontSize: 16,
              },
              values: ['Happy 😊', 'Sad 😞', 'Surprise 😲', 'Neutral 😐', 'Disguist 🤢'],
            },

            series: [
              {
                aspect: 'stepped',
                values: ['Happy 😊', 'Sad 😞', 'Surprise 😲', 'Neutral 😐'],
                text: 'Person 1',
              },
              {
                aspect: 'stepped',
                values: ['Surprise 😲', 'Surprise 😲', 'Happy 😊', 'Sad 😞', 'Disguist 🤢'],
                text: 'Person 2',
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
