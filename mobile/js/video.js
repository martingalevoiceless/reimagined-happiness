import React from 'react'
import {Platform, StyleSheet, View} from 'react-native';
import { UtilComponent } from './util';
import { Player, ControlBar, PlayToggle, PlaybackRateMenuButton, CurrentTimeDisplay, TimeDivider, DurationDisplay, ProgressControl, RemainingTimeDisplay, FullscreenToggle, BigPlayButton } from 'video-react';
import 'video-react/dist/video-react.css';

class Video extends UtilComponent {
    constructor(props) {
        super(props);
        this.vref = React.createRef();
    }
    componentDidMount() {
        console.log(this.vref);
    }
    componentDidUpdate() {
        console.log(this.vref);
    }
    timejump() {
        if (!this.vref.current) { return; }
        var player = (this.vref.current.getState() || {}).player;
        if (!player) { return; }
        var submin1 = Math.max(this.props.min_time-4, 0);
        var submin2 = Math.max(this.props.min_time-5, 0);
        if (this.props.max_time !== undefined && (player.currentTime > this.props.max_time || player.currentTime < submin2)) {
            this.vref.current.seek(submin1)
        }

    }
    render({source, style, autoplay=true, short_controls=true, long_controls=false, onClick, pad=90, min_time, max_time}) {
        return <Player
            playsInline
            style={StyleSheet.flatten([style, {height: '100%', width: '100%', overflow: 'hidden', display: 'inline-block'}])}
            loop={true}
            fluid={false}
            muted
            startTime={min_time && Math.max(min_time-4, 0)}
            autoPlay={autoplay ? true : undefined}
            src={source}
            onPause={event => short_controls ? this.vref.current.play() : null}
            onTimeUpdate={event => this.timejump()}
            ref={this.vref}
        >
            <style></style>
            <BigPlayButton style={{display: 'none'}}/>
            {long_controls && <ControlBar autoHide={false} disableDefaultControls>
                <div className="video-react-control" style={{width: '' + pad + 'px'}}>&nbsp;</div>
                <PlayToggle />
                <CurrentTimeDisplay />
                <TimeDivider />
                <DurationDisplay/>
                <ProgressControl/>
                <RemainingTimeDisplay/>

                <PlaybackRateMenuButton rates={[5, 2, 1, 0.5, 0.1]} />
                <FullscreenToggle/>
                <div className="video-react-control" style={{width: '' + pad + 'px'}}>&nbsp;</div>
            </ControlBar>}
            {short_controls && <ControlBar autoHide={false} disableDefaultControls>
                <div className="video-react-control" style={{width: '' + pad + 'px'}}>&nbsp;</div>
                <CurrentTimeDisplay />
                <TimeDivider />
                <DurationDisplay/>
                <div style={{flexGrow: 1}}>&nbsp;</div>
                <PlaybackRateMenuButton rates={[5, 2, 1, 0.5, 0.1]} />
                <FullscreenToggle/>
                <div className="video-react-control" style={{width: '' + pad + 'px'}}>&nbsp;</div>
            </ControlBar>}
        </Player> 
    }
}

export default Video;
