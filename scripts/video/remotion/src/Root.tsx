import { Composition } from "remotion";
import { Storybook } from "./Storybook";
import scriptData from "./data.json";
import type { ScriptData } from "./types";

const FPS = 30;
const data = scriptData as ScriptData;
const totalDuration = Math.ceil(data.audioDuration * FPS);

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="Storybook"
        component={Storybook}
        durationInFrames={totalDuration}
        fps={FPS}
        width={1920}
        height={1080}
        defaultProps={{ data }}
      />
    </>
  );
};
