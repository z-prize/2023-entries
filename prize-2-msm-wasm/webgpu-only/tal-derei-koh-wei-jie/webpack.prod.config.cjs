/* eslint-disable @typescript-eslint/no-var-requires */
const path = require("path");
const webpack = require("webpack");
const CopyWebpackPlugin = require('copy-webpack-plugin');
const HtmlWebpackPlugin = require("html-webpack-plugin");
const ForkTsCheckerWebpackPlugin = require("fork-ts-checker-webpack-plugin");
const cleanWebpackPlugin = require("clean-webpack-plugin");

const config = {
  mode: "production",
  entry: "./src/index.tsx",
  devServer: {
    client: {
      overlay: false,
    },
  },
  experiments: {
    asyncWebAssembly: true
  },
  output: {
    path: path.resolve(__dirname, "build"),
    filename: "[name].[contenthash].js",
    publicPath: "",
  },
  module: {
    rules: [
      {
        test: /\.css$/i,
        exclude: /node_modules/,
        use: ['style-loader', 'css-loader', 'postcss-loader'],
      },
      {
        test: /\.wgsl/i,
        type: 'asset/source',
      },
      {
        test: /\.(ts|js)x?$/i,
        exclude: /node_modules/,
        use: {
          loader: "babel-loader",
          options: {
            presets: [
              "@babel/preset-env",
              "@babel/preset-react",
              "@babel/preset-typescript",
            ],
          },
        },
      },
    ],
  },
  resolve: {
    extensions: [".tsx", ".ts", ".js"],
    fallback: {
      "crypto": require.resolve("crypto-browserify"),
      "stream": require.resolve("stream-browserify"),
      "buffer": require.resolve("buffer/")
    }
  },
  plugins: [
    new webpack.ProvidePlugin({
      Buffer: ['buffer', 'Buffer'],
    }),
    new HtmlWebpackPlugin({
      template: "public/index.html",
    }),
    new ForkTsCheckerWebpackPlugin({
      async: false,
    }),
    new CopyWebpackPlugin({
      patterns: [
        {
          from: path.resolve(__dirname, 'public/test-data'),
          to: path.resolve(__dirname, 'dist/test-data'),
        }
      ]
    }),
    new cleanWebpackPlugin.CleanWebpackPlugin(),
    new webpack.DefinePlugin({
      'process.env.NODE_ENV': JSON.stringify('production'),
      'process.env.NODE_DEBUG': JSON.stringify(process.env.NODE_DEBUG),
      'process.type': JSON.stringify(process.type),
      'process.version': JSON.stringify(process.version),
    })
  ],
};

const workerConfig = {
  mode: "production",
  cache: {
    type: 'filesystem',
    allowCollectingMemory: true
  },
  performance: {
    hints: false
  },
  experiments: {
    asyncWebAssembly: true,
    syncWebAssembly: true,
    topLevelAwait: true
  },
  target: 'webworker',
  entry: {
    wasmMSM: './src/workers/wasmMSM.ts',
  },
  output: {
    pathinfo: false,
    publicPath: '/',
  },
  resolve: {
    extensions: ['.ts', '.tsx', '.js', '.wasm'],
    alias: {
      shared: path.resolve(__dirname, 'src', 'shared')
    },
    fallback: {
      url: false,
      os: false,
      path: false,
      stream: false,
      crypto: require.resolve("crypto-browserify"),
      http: false,
      https: false,
      buffer: require.resolve('buffer'),
      stream: require.resolve('stream-browserify'),
      assert: require.resolve('assert')
    }
  },
  plugins: [
    new webpack.ProvidePlugin({
      Buffer: ['buffer', 'Buffer']
    }),

    new webpack.ProvidePlugin({
      process: 'process/browser'
    }),
  ],
  module: {
    rules: [
      {
        test: /\.m?js$/i,
        exclude: /node_modules/,
        type: 'javascript/auto'
      },
      {
        test: /\.(ts|js)x?$/i,
        exclude: /node_modules/,
        use: {
          loader: "babel-loader",
          options: {
            presets: [
              "@babel/preset-env",
              "@babel/preset-typescript",
            ],
          },
        },
      },
    ]
  }
};


module.exports = [config, workerConfig];
